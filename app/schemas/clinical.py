from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, SQLModel

FEATURE_COLUMNS = [
    "GENDER",
    "WBC",
    "NE",
    "LY",
    "MO",
    "EO",
    "RBC",
    "HGB",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
    "RDW",
    "PLT",
    "FERRITIN",
    "MENTZER_INDEX",
]

PATHOLOGY_COLS = [
    "LEUKEMIA_SUSPECT",
    "VIRAL_INF",
    "BACTERIAL_INF",
    "MONOCYTOSIS",
    "EOSINOPHILIA",
    "CHRONIC_DISEASE_ANEMIA",
    "THALASSEMIA_TRAIT",
    "IRON_DEFICIENCY_ANEMIA",
    "THROMBOCYTOPENIA",
    "THROMBOCYTOSIS",
]


class HematologyInput(BaseModel):
    GENDER: int = Field(..., description="1: Male, 0: Female")
    WBC: float = Field(..., description="White blood cell count")
    NE: float = Field(..., description="Neutrophils")
    LY: float = Field(..., description="Lymphocytes")
    MO: float = Field(..., description="Monocytes")
    EO: float = Field(..., description="Eosinophils")
    RBC: float = Field(..., description="Red blood cell count")
    HGB: float = Field(..., description="Hemoglobin")
    HCT: float = Field(..., description="Hematocrit")
    MCV: float = Field(..., description="Mean corpuscular volume")
    MCH: float = Field(..., description="Mean corpuscular hemoglobin")
    MCHC: float = Field(..., description="Mean corpuscular hemoglobin concentration")
    RDW: float = Field(..., description="Red cell distribution width")
    PLT: float = Field(..., description="Platelet count")
    FERRITIN: float = Field(..., description="Ferritin level")
    MENTZER_INDEX: float = Field(..., description="Calculated from MCV and RBC")
    GENDER_LABEL: Literal["Male", "Female"] | None = Field(
        default=None, description="Optional label for logging/querying; not passed to model"
    )


class PredictionResponse(BaseModel):
    predictions: dict[str, int] = Field(
        ..., description="Binary prediction (0 or 1) for each of the 10 pathology columns"
    )
    probabilities: dict[str, float] | None = Field(
        default=None, description="Probability of class 1 for each pathology column"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class PredictionRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))

    gender: int
    gender_label: str | None = None
    wbc: float
    ne: float
    ly: float
    mo: float
    eo: float
    rbc: float
    hgb: float
    hct: float
    mcv: float
    mch: float
    mchc: float
    rdw: float
    plt: float
    ferritin: float
    mentzer_index: float

    leukemia_suspect: int
    viral_inf: int
    bacterial_inf: int
    monocytosis: int
    eosinophilia: int
    chronic_disease_anemia: int
    thalassemia_trait: int
    iron_deficiency_anemia: int
    thrombocytopenia: int
    thrombocytosis: int