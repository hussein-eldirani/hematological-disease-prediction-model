import random
import time
import httpx

API_URL = "http://127.0.0.1:8000/predict"

METRIC_RANGES = {
    "WBC": (0.50, 250.00),
    "NE": (0.00, 200.00),
    "LY": (0.00, 90.00),
    "MO": (0.00, 20.00),
    "EO": (0.00, 10.00),
    "RBC": (0.80, 7.50),
    "HGB": (2.90, 23.00),
    "HCT": (7.00, 68.00),
    "MCV": (49.90, 125.30),
    "MCH": (12.50, 139.00),
    "MCHC": (18.20, 168.00),
    "RDW": (9.70, 39.40),
    "PLT": (13.00, 1770.00),
    "FERRITIN": (0.50, 28000.00),
}


def _generate_wbc_and_differentials() -> dict[str, float]:
    """Generates WBC in [0.50, 250.00] and splits it across NE, LY, MO, EO."""
    wbc_min_units = int(round(METRIC_RANGES["WBC"][0] * 100))
    wbc_max_units = int(round(METRIC_RANGES["WBC"][1] * 100))
    wbc_units = random.randint(wbc_min_units, wbc_max_units)

    diff_caps = [
        ("NE", int(round(METRIC_RANGES["NE"][1] * 100))),
        ("LY", int(round(METRIC_RANGES["LY"][1] * 100))),
        ("MO", int(round(METRIC_RANGES["MO"][1] * 100))),
        ("EO", int(round(METRIC_RANGES["EO"][1] * 100))),
    ]
    random.shuffle(diff_caps)

    remaining = wbc_units
    allocated: dict[str, int] = {}

    for idx, (name, cap) in enumerate(diff_caps):
        if idx == len(diff_caps) - 1:
            allocated[name] = remaining
        else:
            other_max = sum(c for _, c in diff_caps[idx + 1 :])
            min_take = max(0, remaining - other_max)
            max_take = min(cap, remaining)
            take = random.randint(min_take, max_take)
            allocated[name] = take
            remaining -= take

    return {
        "WBC": round(wbc_units / 100.0, 2),
        "NE": round(allocated["NE"] / 100.0, 2),
        "LY": round(allocated["LY"] / 100.0, 2),
        "MO": round(allocated["MO"] / 100.0, 2),
        "EO": round(allocated["EO"] / 100.0, 2),
    }


def generate_patient_payload() -> dict:
    """Generates a single unlabeled hematology feature payload."""
    gender = random.choice([0, 1])  # 1: Male, 0: Female
    gender_label = "Male" if gender == 1 else "Female"

    diffs = _generate_wbc_and_differentials()

    rbc = round(random.uniform(*METRIC_RANGES["RBC"]), 2)
    mcv = round(random.uniform(*METRIC_RANGES["MCV"]), 2)
    mentzer_index = round(mcv / rbc, 2) if rbc > 0 else 0.0

    # Keep HGB slightly below 23.00 upper bound so 3 * HGB fits cleanly inside HCT [7.00, 68.00]
    hgb_max_for_rule = min(METRIC_RANGES["HGB"][1], METRIC_RANGES["HCT"][1] / 3.0)
    hgb = round(random.uniform(METRIC_RANGES["HGB"][0], hgb_max_for_rule), 2)
    hct_raw = (3.0 * hgb) + random.uniform(-1.2, 1.2)
    hct = round(
        max(METRIC_RANGES["HCT"][0], min(METRIC_RANGES["HCT"][1], hct_raw)),
        2,
    )

    return {
        "GENDER": gender,
        **diffs,
        "RBC": rbc,
        "HGB": hgb,
        "HCT": hct,
        "MCV": mcv,
        "MCH": round(random.uniform(*METRIC_RANGES["MCH"]), 2),
        "MCHC": round(random.uniform(*METRIC_RANGES["MCHC"]), 2),
        "RDW": round(random.uniform(*METRIC_RANGES["RDW"]), 2),
        "PLT": round(random.uniform(*METRIC_RANGES["PLT"]), 2),
        "FERRITIN": round(random.uniform(*METRIC_RANGES["FERRITIN"]), 2),
        "MENTZER_INDEX": mentzer_index,
        "GENDER_LABEL": gender_label,
    }


def simulate_lab_traffic(num_requests: int = 10, delay_seconds: float = 1.5) -> None:
    """Sends simulated unlabeled patient samples to the FastAPI /predict endpoint."""
    with httpx.Client(timeout=10.0) as client:
        for i in range(num_requests):
            sample = generate_patient_payload()
            try:
                response = client.post(API_URL, json=sample)
                print(
                    f"[{i + 1}/{num_requests}] Status {response.status_code}: {response.json()}"
                )
            except httpx.RequestError as exc:
                print(
                    f"[{i + 1}/{num_requests}] Could not reach API ({API_URL}): {exc}"
                )
            time.sleep(delay_seconds)


if __name__ == "__main__":
    simulate_lab_traffic()