# 🛍️  Customer-Repurchase Predictor

End-to-end machine-learning pipeline that estimates **the probability a first-time
shopper will buy again within seven days**.

| Layer / Concern        | Technology & Libraries                           |
|------------------------|---------------------------------------------------|
| Feature engineering    | `pandas` · custom RFM / session aggregates       |
| Model & imbalance      | `XGBoost` (hist) + `SMOTE` (`imblearn`)           |
| Experiment tracking    | **MLflow** (local file store)                    |
| Explainability         | **SHAP** summary & dependence plots              |
| Reproducibility        | `requirements.txt` + `environment.yml`           |

---

## 🚀 Quick start

> All commands assume you are at the **repo root**.  
> Windows ⇢ back-slashes \
> · macOS/Linux ⇢ forward-slashes /

```bash
# 1️⃣  create & activate an isolated environment
python -m venv .venv && .\.venv\Scripts\activate          # Windows
# source .venv/bin/activate                               # macOS/Linux
pip install -r requirements.txt

# 2️⃣  train + register the pipeline
python -m ubp.train --model-dir models

# 3️⃣  generate a SHAP feature-importance plot
python scripts/shap_report.py                             # → reports/feature_importance.png 

Start the MLflow UI in another terminal: mlflow ui --port 5000
Then browse to http://127.0.0.1:5000 to inspect parameters, metrics and artefacts.

🎯 Key results <small>(reference run)</small>

🗂️ Project layout
├─ data/                     raw & sample CSVs (git‑ignored)
├─ mlruns/                   MLflow experiment runs & registry
├─ models/                   local fallback pickle
├─ notebooks/                exploratory notebooks
├─ reports/                  SHAP + screenshot assets
├─ scripts/
│   ├─ shap_report.py        SHAP summary PNG generator
│   └─ predict.py            batch scorer (registry or local)
└─ src/
    └─ ubp/
        ├─ data.py           load_dataset()
        ├─ features.py       build_feature_table()
        ├─ pipeline.py       pipeline helpers
        ├─ train.py          executed via `python -m ubp.train`
        └─ __init__.py

🛠️ Local development
pytest -q                 # run smoke tests
mlflow ui --port 5000     # open experiment dashboard

📜 Licence & credits
Released under the MIT Licence.
Dataset courtesy of <UCI Online Retail II Dataset https://archive.ics.uci.edu/dataset/502/online+retail+ii>.
Project by Vicnent Nguyen – connect on LinkedIn.