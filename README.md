<div align="center">

<br/>

```
██████╗ ██╗      █████╗  ██████╗███████╗███╗   ███╗███████╗███╗   ██╗████████╗
██╔══██╗██║     ██╔══██╗██╔════╝██╔════╝████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
██████╔╝██║     ███████║██║     █████╗  ██╔████╔██║█████╗  ██╔██╗ ██║   ██║   
██╔═══╝ ██║     ██╔══██║██║     ██╔══╝  ██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   
██║     ███████╗██║  ██║╚██████╗███████╗██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   
╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   
                                                                                 
                T I E R   P R E D I C T O R
```

<br/>

<!-- Badges -->
<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Active-00C853?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>

<br/><br/>

> **An intelligent ML system that classifies engineering students into placement tiers — Regular, Dream, or Super Dream — using Random Forest Classification on multi-dimensional academic and co-curricular profiles.**

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Model Performance](#-model-performance)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Sample Output](#-sample-output)
- [Feature Engineering Details](#-feature-engineering-details)
- [Future Roadmap](#-future-roadmap)

---

## 🎯 Overview

The **Placement Tier Predictor** is a capstone project built for the AIML curriculum. It ingests raw student data — grades, coding scores, internship history, projects, and more — and predicts which **placement tier** a student is likely to fall into:

| Tier | Description |
|------|-------------|
| 🟢 **Regular** | Entry-level packages; core placements |
| 🟡 **Dream** | Mid-to-high packages; competitive roles |
| 🔴 **Super Dream** | Top-tier packages; elite company offers |

Tier boundaries are determined **dynamically** using quantile-based binning (`pd.qcut`) on real CTC data, ensuring fair and data-driven categorization regardless of dataset scale.

---

## ✨ Key Features

- 🧹 **Robust Data Cleaning** — Handles unit-suffixed strings (`85 %`), null values, and malformed entries gracefully
- 🔧 **Smart Feature Engineering** — Extracts branch from student IDs, computes seniority years, and ordinally encodes internship levels
- 🌲 **Random Forest Classifier** — Ensemble model with 100 estimators and depth control for high accuracy with low overfitting
- 📊 **Dual Evaluation Context** — Classification metrics (Accuracy, F1) *and* regression metrics (RMSE, R²) for comprehensive model assessment
- 🔮 **Live Inference Function** — `predict_student_tier()` accepts raw student dictionaries and returns real-time predictions
- 📈 **Confusion Matrix Visualization** — Heatmap output to diagnose inter-tier misclassification patterns

---

## 🛠️ Tech Stack

| Category | Library / Tool |
|----------|---------------|
| Language | Python 3.9+ |
| Data Processing | `pandas`, `numpy` |
| Machine Learning | `scikit-learn` |
| Visualization | `matplotlib`, `seaborn` |
| Model | `RandomForestClassifier` |
| Preprocessing | `StandardScaler`, `pd.get_dummies` |

---

## 🏗️ Project Architecture

```
Placement_Data.csv
        │
        ▼
┌───────────────────┐
│  1. Data Loading  │  ── CSV ingestion with error handling
└────────┬──────────┘
         │
         ▼
┌───────────────────────┐
│  2. Data Cleaning     │  ── Unit stripping, null imputation (median)
└────────┬──────────────┘
         │
         ▼
┌───────────────────────┐
│  3. Feature           │  ── Branch extraction, seniority calc,
│     Engineering       │     internship ordinal encoding
└────────┬──────────────┘
         │
         ▼
┌───────────────────────┐
│  4. Target Creation   │  ── qcut → Regular / Dream / Super Dream
└────────┬──────────────┘
         │
         ▼
┌───────────────────────┐
│  5. Encoding &        │  ── One-Hot (branch) + StandardScaler
│     Scaling           │     (numerical features)
└────────┬──────────────┘
         │
         ▼
┌───────────────────────┐
│  6. Model Training    │  ── RandomForestClassifier (n=100, depth=15)
└────────┬──────────────┘
         │
         ▼
┌───────────────────────┐
│  7. Evaluation        │  ── Accuracy, F1, RMSE, R² + Confusion Matrix
└────────┬──────────────┘
         │
         ▼
┌───────────────────────┐
│  8. Inference         │  ── predict_student_tier() live predictions
└───────────────────────┘
```

---

## 🔍 Pipeline Walkthrough

### Step 1 — Data Loading
The dataset `Placement_Data.csv` is loaded via `pandas`. A `try-except` block ensures graceful failure with a descriptive error if the file is missing.

### Step 2 — Data Cleaning & Preprocessing
The `clean_and_convert()` utility:
- Strips unit strings (e.g., `%` from aptitude scores)
- Converts columns to numeric types using `pd.to_numeric(..., errors='coerce')`
- Replaces missing values with the **column median** to preserve the underlying data distribution

### Step 3 — Feature Engineering
| Transformation | Input Column | Output Column |
|---|---|---|
| String split | `student_id` | `branch` |
| Year delta | `year_of_joining` | `seniority_years` |
| Ordinal map | `internships` (text) | `internships` (0–3) |

Internship mapping:
```python
{'None': 0, 'First Internship': 1, 'Second Internship': 2, 'Third & Above': 3}
```

### Step 4 — Target Transformation
`package_lpa` is binned into 3 equal-frequency quantile buckets and labelled as the classification target. The original continuous column is preserved separately for the regression evaluation context.

### Step 5 — Encoding & Scaling
- **One-Hot Encoding**: Applied to `branch` with `drop_first=True` to avoid multicollinearity
- **StandardScaler**: Applied to all 7 numerical features, ensuring zero mean and unit variance so no single feature dominates the ensemble

### Step 6 — Model Training
```python
RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
```
- `n_jobs=-1` enables full CPU parallelism during training
- `max_depth=15` prevents overfitting on noisy student records
- 80/20 train-test split with `random_state=42` for reproducibility

---

## 📊 Model Performance

```
══════════════════════════════════════════════════
          PROJECT EVALUATION REPORT
══════════════════════════════════════════════════
  Accuracy Score   ████████████████████  ~XX.XX%
  F1 Score (Wt.)   ████████████████████  ~X.XXXX
  R-squared (R²)   ████████████████████  ~X.XXXX
  RMSE (Error)     ₹X.XX LPA
──────────────────────────────────────────────────
```

> ℹ️ *Exact metric values are populated at runtime from your dataset. The confusion matrix heatmap provides visual breakdown of per-tier prediction accuracy.*

**Confusion Matrix (Illustrative Structure):**

```
              Predicted
              Regular │ Dream │ Super Dream
             ─────────┼───────┼────────────
Actual Regular │  TP   │  FP  │     FP
         Dream │  FN   │  TP  │     FP
   Super Dream │  FN   │  FN  │     TP
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/placement-tier-predictor.git
cd placement-tier-predictor
```

### 2. Create a Virtual Environment *(recommended)*
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

### 4. Add Your Dataset
Place your `Placement_Data.csv` file in the project root directory.

**Expected CSV Schema:**

| Column | Type | Example |
|--------|------|---------|
| `student_id` | string | `STUDENT_001 CSE Core` |
| `cgpa` | float | `8.7` |
| `coding_score` | int | `620` |
| `aptitude_score` | string | `78 %` |
| `projects` | int | `3` |
| `internships` | string | `Second Internship` |
| `year_of_joining` | int | `2022` |
| `backlogs` | int | `0` |
| `package_lpa` | float | `12.5` |

---

## 🚀 Usage

### Run the Full Pipeline
```bash
python placement_predictor.py
```

### Live Inference on a Custom Student Profile
```python
sample_student = {
    'student_id': 'STUDENT_999 CSE Core',
    'cgpa': 9.2,
    'coding_score': 650,
    'aptitude_score': '85 %',
    'projects': 4,
    'internships': 'Second Internship',
    'year_of_joining': 2022,
    'backlogs': 0
}

result = predict_student_tier(clf_model, sample_student)
print(f"Predicted Placement Tier: {result}")
```

---

## 🖥️ Sample Output

```
══════════════════════════════════════════════════
              PROJECT EVALUATION REPORT
══════════════════════════════════════════════════
Accuracy Score:    0.9150
F1 Score (Weight): 0.9132
R-squared (R2):    0.8874
RMSE (Error):      ₹1.23 LPA
──────────────────────────────────────────────────
SIMULATION RESULT:
Profile: CGPA 9.2, CSE Core, 2 Internships
Predicted Placement Tier: Super Dream
══════════════════════════════════════════════════
```
*(Values are illustrative — actual output depends on your dataset)*

---

## 🔬 Feature Engineering Details

| Feature | Source | Transformation | Rationale |
|---------|--------|---------------|-----------|
| `branch` | `student_id` (split) | One-Hot Encoded | Different branches have different placement rates |
| `seniority_years` | `year_of_joining` | `2026 - year` | Proxy for experience and exposure |
| `internships` | Text category | Ordinal (0–3) | Progressive internship count has linear value |
| `aptitude_score` | String (`"85 %"`) | Strip → float | Raw data had unit noise |
| `cgpa`, `coding_score` | Numeric | StandardScaler | Ensures equal feature weightage in RF |

---

## 🗺️ Future Roadmap

- [ ] **Hyperparameter Tuning** — GridSearchCV / RandomizedSearchCV for optimal `n_estimators`, `max_depth`
- [ ] **SHAP Explainability** — Feature importance visualization per prediction
- [ ] **Streamlit Dashboard** — Interactive web UI for college counselors and students
- [ ] **Multi-label Support** — Predict multiple target companies alongside tier
- [ ] **Cross-Validation** — k-fold CV for more robust performance estimates
- [ ] **Model Persistence** — `joblib` serialization for deployment

---

<div align="center">

---

**Built with 🤖 ML and ☕ by an AIML Capstone Team**

*If this project helped you, consider starring ⭐ the repository!*

</div>
