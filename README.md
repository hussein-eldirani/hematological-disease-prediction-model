# Multi-Label Hematological Disease Prediction via Weak Supervision

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.5.3-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.6-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-2B5B84?style=for-the-badge&logo=xgboost&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.18.1-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.11.2-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-3776AB?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter_Lab-IPykernel_7.4-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

A end-to-end Machine Learning pipeline and clinical diagnostic system engineered to detect 10 parallel hematological pathologies from raw Complete Blood Count (CBC) vectors.

Lacking structured ground-truth clinical diagnoses in raw lab records, this project deploys a Guideline-Driven Weak Supervision Rule Engine to programmatically generate 10 binary target classes permitting co-occurring pathologies. Baseline classification is performed using a MultiOutputClassifier wrapped around regularized XGBoost decision trees, evaluated against multi-label evaluation metrics (Hamming Loss, Per-Label Recall/Precision).

## Data Source & Disclaimer

* **Origin:** Sourced from an open-access Kaggle dataset (~4 MB) containing raw Complete Blood Count (CBC) parameters.
* **Unstructured & Undocumented Raw Data:** The raw file lacked clinical ground-truth labels, feature documentation, and explicit measurement units.
* **Demographic & Unit Standardization:**
  * **Gender Mapping:** Reconstructed biological sex labels by analyzing feature distributions and standard clinical reference range splits.
  * **Global Unit Harmonization:** Evaluated parameter value ranges against standard international clinical bounds to systematically determine and harmonize measurement units across all lab features (e.g., $10^3/\mu\text{L}$ for leukocytes, $\text{g/dL}$ for hemoglobin, $\text{fL}$ for MCV).
* **Clinical Target Engineering:** All 10 pathology targets were programmatically derived from the standardized raw parameters (`WBC`, `RBC`, `HGB`, `HCT`, `MCV`, `MCH`, `MCHC`, `RDW`, `PLT`, `NE`, `LY`, `MO`, `EO`) using hematological guidelines and weak supervision in `notebooks/01_EDA.ipynb`.

> **Disclaimer:** This project is developed strictly for educational, portfolio, and analytical demonstration purposes. It does not constitute formal clinical software or medical diagnostic advice.

## Project Structure

```text
├── data/
│   ├── raw/                  # Original raw CBC lab records
│   └── processed/            # Cleaned, Rule-of-Three validated dataset (18,081 samples)
├── models/
│   └── multi_label_disease_model.joblib # Trained XGBoost MultiOutput pipeline
├── notebooks/
│   ├── 01_eda.ipynb          # Preprocessing, Gender Decoding, Quality Gate & Labeling
│   └── 02_training.ipynb     # Multi-Label XGBoost Training & Error Analysis
├── .gitignore                # Environment, model, and data exclusions
├── requirements.txt          # Pinned dependency requirements
└── README.md                 # Project documentation
```

## Executive Summary & Architectural Overview

1. Raw Clinical Vector (CBC Lab Records)
2. Biological Quality Gate (Rule of Three Validation)
3. Clinical Feature Engineering (Mentzer Index, Gender)
4. Weak Supervision Engine (10 Guideline Pathways) 
5. MultiOutput XGBoost Classifier (Subsample & Depth Constrained)
6. Multi-Label Predictions (10 Pathology Outputs)

## Exploratory Data Analysis & Clinical Preprocessing

1. **Feature Hygiene & Vectorization**

* **Noise Truncation:** Dropped hardware-specific/machine-variance metrics (MPV, PCT, PDW, SD, SDTSD, TSD) and unsegmented differential flags (BA#) to reduce feature noise and prevent decision trees from memorizing instrument artifacts.

* **XGBoost Compatibility:** Normalized column headers (stripping special characters #) to ensure clean JSON booster parsing during model training.

* **Vectorized Mentzer Index (O(N)):** Engineered the clinical ratio $\text{Mentzer Index} = \frac{\text{MCV}}{\text{RBC}}$ using vectorized array logic (np.where), explicitly handling zero-division edge cases to help decision trees delineate microcytic anemias ($\text{Mentzer} < 13 \rightarrow \text{Thalassemia Trait}$ vs. $> 13 \rightarrow \text{Iron Deficiency Anemia}$).

2. **Programmatic Demographic Inference**

Raw unmapped binary gender flags were decoded by leveraging physiological baselines. Group medians across testosterone-driven erythroid metrics ($\text{HGB}, \text{RBC}, \text{HCT}$) were computed programmatically to map higher erythroid mass to male subjects.

3. **Biological Data Quality Gate: The "Rule of Three"**

To protect model integrity against pre-analytical lab errors (such as hemolyzed tubes, diluted samples, or clotted specimens), a biological validation gate was introduced:

$$\text{HCT (\%)} \approx 3 \times \text{HGB (g/dL)}$$

Samples exhibiting an absolute deviation $> 6\%$ were flagged as measurement anomalies and purged.

* **Anomalies Purged:** 11,919 samples (39.73%)
* **Clean Dataset Retained:** 18,081 samples (60.27%)

## Weak Supervision & Target Labeling Engine

Lacking structured clinical diagnoses, a deterministic clinical knowledge engine derived from hematological guidelines was deployed to map patient vectors across three primary physiological tracks:

1. **Leukocytic Track:** Identifies acute leukemias ($\text{WBC} > 50$), differential-driven acute infections ($\text{Bacterial}$ vs. $\text{Viral}$ via Neutrophil/Lymphocyte dominance), $\text{Eosinophilia}$, and $\text{Monocytosis}$.

2. **Erythrocytic Track:** Evaluates gender-stratified anemia baselines ($\text{HGB} < 13.5 \text{ Male} \mid < 12.0 \text{ Female}$), splitting microcytic branches into $\text{Thalassemia Trait}$, $\text{Iron Deficiency Anemia (IDA)}$, and $\text{Anemia of Chronic Disease (ACD)}$ via Mentzer Index, RDW, and Ferritin cutoffs.

3. **Thrombocytic Track:** Categorizes quantitative platelet disorders ($\text{Thrombocytosis} > 450 \text{ k/µL}$ vs. $\text{Thrombocytopenia} < 150 \text{ k/µL}$).

## Model Architecture & Baseline Performance

To evaluate how effectively standard machine learning algorithms learn non-linear clinical pathways from 14 raw CBC parameters (WBC, RBC, HGB, HCT, MCV, MCH, MCHC, RDW, PLT, NE, LY, MO, EO, GENDER), an regularized XGBoost model wrapped inside a MultiOutputClassifier was trained on a 60/40 train/test split ($N_{\text{test}} = 7,233$).

1. **Key Metrics Summary** Overall Multi-Label Hamming Loss: 0.0058 (99.42% global label accuracy)

2. **Key Performance Insights**

* **Deterministic Sharpness on Single-Threshold Target Classes (F1 = 0.99 – 1.00):** Pathologies governed by explicit quantitative bounds on primary CBC markers—such as LEUKEMIA_SUSPECT ($\text{WBC} > 50$), THROMBOCYTOSIS ($\text{PLT} > 450$), and absolute differential counts (MONOCYTOSIS, EOSINOPHILIA)—achieved 1.00 F1-scores. This confirms that XGBoost tree splits can cleanly reverse-engineer sharp numeric thresholds without decision boundary degradation across parallel trees.

* **Differential Ratio Recovery (BACTERIAL_INF & VIRAL_INF):** The model achieved 0.99 F1 on BACTERIAL_INF and 0.95 F1 on VIRAL_INF. This demonstrates that decision tree ensembles successfully learn multi-feature relative percentages (Neutrophil vs. Lymphocyte dominance) directly from raw absolute leukocyte components.

* **Anemia Overlap & Diagnostic Trade-offs (IDA vs. ACD):**  **Iron Deficiency Anemia (IDA):** Precision = 0.80, Recall = 0.86, F1 = 0.83 / **Anemia of Chronic Disease (ACD):** Precision = 0.82, Recall = 0.58, F1 = 0.68. The clinical root cause of this result is that secondary confirmatory assays (e.g., Serum Ferritin, Total Iron Binding Capacity) were excluded from feature matrix $X$ to simulate low-cost primary screening, the model must differentiate microcytic/normocytic anemias solely using $HGB$, $MCV$, and $RDW$. The model exhibits false-positive bleed between IDA and ACD, causing a recall drop in ACD (58%) as subtle cases are misclassified as IDA.

### Quickstart & Setup

1. **Clone & Environment Setup**

```
git clone https://github.com/hussein-eldirani/hematological-disease-prediction-model
cd hematological-disease-prediction-model

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Model Inference Example**

```
import joblib
import pandas as pd

# Load saved multi-label model
model = joblib.load("models/multi_label_disease_model.joblib")

# Example raw patient CBC vector
sample_patient = pd.DataFrame([{
    'WBC': 14.2, 'RBC': 3.8, 'HGB': 10.1, 'HCT': 31.0, 
    'MCV': 72.0, 'MCH': 24.0, 'MCHC': 31.0, 'RDW': 16.2, 
    'PLT': 480, 'NE': 9.1, 'LY': 3.2, 'MO': 1.1, 'EO': 0.2, 'GENDER': 1
}])

# Generate predictions across all 10 pathology targets
predictions = model.predict(sample_patient)
print("Predicted Pathology Vector:", predictions)
```
