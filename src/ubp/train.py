#!/usr/bin/env python
# scripts/train.py
# ------------------------------------------------------------
# Fit an XGB + SMOTE pipeline, log everything to MLflow,
# and persist the best model to models/xgb_smote_pipeline.pkl
# ------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
import warnings

import click
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline      import Pipeline as ImbPipeline
from sklearn.compose        import ColumnTransformer
from sklearn.metrics        import (average_precision_score,
                                    classification_report,
                                    precision_recall_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing   import StandardScaler
from xgboost                 import XGBClassifier

# ── MLflow ---------------------------------------------------
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# 1) Point MLflow to repo-local  mlruns/  (portable across shells)
TRACKING_URI = (Path(__file__).resolve().parents[2] / "mlruns").as_uri()
mlflow.set_tracking_uri(TRACKING_URI)

# 2) Ensure the experiment exists (creates if missing)
mlflow.set_experiment("repurchase_xgb")

# 3) Enable autologging
mlflow.autolog(
    disable=False,
    exclusive=False,
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True,
)

# ── project modules -----------------------------------------
from ubp import data          # load_dataset()
from ubp import features as ft   # build_feature_table()

warnings.filterwarnings("ignore", category=UserWarning)


@click.command(context_settings=dict(show_default=True))
@click.option("--model-dir",
              default="models",
              show_default=True,
              help="Folder that will receive xgb_smote_pipeline.pkl")
def main(model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣  Data ------------------------------------------------------------------
    dataset = data.load_dataset()
    leak_cols = ["days_to_repurchase", "recency_2nd",
                 "diversity_7d", "freq_3d", "monetary_3d"]

    X = (dataset
         .drop(columns=["CustomerID", "did_repurchase_7d",
                        "first_date", "last_date"] + leak_cols))
    y = dataset["did_repurchase_7d"].fillna(0).astype(int)

    idx_sorted = dataset.sort_values("first_date").index
    X, y = X.loc[idx_sorted], y.loc[idx_sorted]

    tscv = TimeSeriesSplit(n_splits=5, test_size=180)

    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("bool").columns

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", "passthrough",  cat_cols)]
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        learning_rate=0.05,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42,
    )

    pipe = ImbPipeline([
        ("pre",   pre),
        ("smote", SMOTE(random_state=42)),
        ("clf",   xgb)
    ])

    gcv = GridSearchCV(
        pipe,
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__max_depth":    [3, 4],
        },
        cv=tscv,
        scoring="average_precision",
        n_jobs=-1,
        verbose=1,
    )

    # 2️⃣  Train & log -----------------------------------------------------------
    with mlflow.start_run(run_name="xgb_smote_cv"):
        gcv.fit(X, y)
        best_pipe = gcv.best_estimator_

        print(f"⭐ Best CV PR-AUC : {gcv.best_score_:.3f}")
        print("Best params :", gcv.best_params_)

        train_idx, test_idx = list(tscv.split(X))[-1]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        y_prob = best_pipe.predict_proba(X_test)[:, 1]

        prec, rec, thr = precision_recall_curve(y_test, y_prob)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        thr_opt = thr[f1.argmax()]

        y_pred = (y_prob >= thr_opt).astype(int)
        print(f"Optimal threshold = {thr_opt:.2f}")
        print(classification_report(y_test, y_pred, digits=3))
        holdout_pr_auc = average_precision_score(y_test, y_prob)
        print("Hold-out PR-AUC :", holdout_pr_auc)

        # manual metrics
        mlflow.log_metric("thr_opt", thr_opt)
        mlflow.log_metric("holdout_pr_auc", holdout_pr_auc)

        # 3️⃣  Persist locally ----------------------------------------------------
        out_path = model_dir / "xgb_smote_pipeline.pkl"
        joblib.dump(best_pipe, out_path)
        print(f"💾 Pipeline saved to {out_path}")

        # 4️⃣  Log & register model WITHOUT the deprecated artifact_path ---------
        signature = infer_signature(X_test, best_pipe.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=best_pipe,
            name="model",                      # replaces artifact_path
            registered_model_name="repurchase_xgb_smote",
            signature=signature,
            input_example=X_test.head(3),
        )
        


if __name__ == "__main__":
    main()
