#!/usr/bin/env python
# scripts/shap_report.py
# ------------------------------------------------------------
# Generate a SHAP summary plot for the fitted XGB-SMOTE pipeline
# ------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
import warnings

import click
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# -- project modules
from ubp import data           # load_dataset()
from ubp import features as ft # build_feature_table() – import kept for consistency

# ── Global matplotlib / warning tweaks ───────────────────────
warnings.filterwarnings("ignore", category=UserWarning)  # numba / perf warnings
plt.rcParams["figure.dpi"] = 120


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("models/xgb_smote_pipeline.pkl"),
    help="Path to the fitted pipeline produced by train.py",
)
@click.option(
    "--n-sample",
    type=int,
    default=200,
    help="Number of rows to explain (kept chronologically if possible).",
)
@click.option(
    "--out-png",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("reports/feature_importance.png"),
    help="Destination PNG for the SHAP summary plot.",
)
def main(model_path: Path, n_sample: int, out_png: Path) -> None:
    """
    Re-build the modelling matrix, load the trained artefact (either a full
    sklearn/imblearn Pipeline or a {'pre':…, 'clf':…} dict), compute SHAP
    values and save a summary plot.
    """
    # ---------------------------------------------------------------------
    # 1️⃣  Re-create the feature table exactly as during training
    # ---------------------------------------------------------------------
    dataset = data.load_dataset()  # shape e.g. (4339, 53)
    leak_cols = [
        "days_to_repurchase",
        "recency_2nd",
        "diversity_7d",
        "freq_3d",
        "monetary_3d",
    ]
    X_full = (
        dataset.drop(
            columns=[
                "did_repurchase_7d",
                "first_date",
                "last_date",
                *leak_cols,
            ],
            errors="ignore",
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------------
    # 2️⃣  Choose a sample of rows to explain
    # ---------------------------------------------------------------------
    date_cols = [c for c in X_full.columns if c.endswith("_date")]
    if date_cols:
        sort_col = sorted(date_cols)[0]                 # earliest date column
        X_full = X_full.sort_values(sort_col).reset_index(drop=True)
        X_sample = X_full.tail(min(n_sample, len(X_full)))
    else:
        rng = np.random.default_rng(seed=0)
        take = rng.choice(len(X_full), size=min(n_sample, len(X_full)), replace=False)
        X_sample = X_full.iloc[np.sort(take)].reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 3️⃣  Load the trained artefact (Pipeline *or* dict) and extract parts
    # ---------------------------------------------------------------------
    obj = joblib.load(model_path)

    if isinstance(obj, dict):                     # {'pre': ..., 'clf': ...}
        pre = obj["pre"]
        clf = obj["clf"]
    elif hasattr(obj, "named_steps"):             # sklearn / imblearn Pipeline
        pre = obj.named_steps["pre"]
        clf = obj.named_steps["clf"]
    else:
        raise TypeError(
            f"Unsupported artefact type: {type(obj)}. "
            "Expecting a dict with keys 'pre' and 'clf' or a Pipeline."
        )

    # ---------------------------------------------------------------------
    # 4️⃣  Transform the raw sample using the identical pre-processor
    # ---------------------------------------------------------------------
    X_enc = pre.transform(X_sample)

    # ---------------------------------------------------------------------
    # 5️⃣  Compute SHAP values with TreeExplainer
    # ---------------------------------------------------------------------
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_enc)
    # For binary classification shap_values may be [neg, pos]
    shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    # ---------------------------------------------------------------------
    # 6️⃣  Plot and save summary figure
    # ---------------------------------------------------------------------
    shap.summary_plot(
        shap_vals,
        X_enc,
        feature_names=X_sample.columns,
        show=False,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    print(f"✅ SHAP report saved ➜ {out_png}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
