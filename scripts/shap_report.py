from __future__ import annotations

from pathlib import Path
import warnings

import click
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt


from ubp import data           
from ubp import features as ft 

warnings.filterwarnings("ignore", category=UserWarning)  
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
    
    dataset = data.load_dataset()  
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

    date_cols = [c for c in X_full.columns if c.endswith("_date")]
    if date_cols:
        sort_col = sorted(date_cols)[0]                 # earliest date column
        X_full = X_full.sort_values(sort_col).reset_index(drop=True)
        X_sample = X_full.tail(min(n_sample, len(X_full)))
    else:
        rng = np.random.default_rng(seed=0)
        take = rng.choice(len(X_full), size=min(n_sample, len(X_full)), replace=False)
        X_sample = X_full.iloc[np.sort(take)].reset_index(drop=True)

    obj = joblib.load(model_path)

    if isinstance(obj, dict):                  
        pre = obj["pre"]
        clf = obj["clf"]
    elif hasattr(obj, "named_steps"):             
        pre = obj.named_steps["pre"]
        clf = obj.named_steps["clf"]
    else:
        raise TypeError(
            f"Unsupported artefact type: {type(obj)}. "
            "Expecting a dict with keys 'pre' and 'clf' or a Pipeline."
        )

    X_enc = pre.transform(X_sample)

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_enc)
    # For binary classification shap_values may be [neg, pos]
    shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    shap.summary_plot(
        shap_vals,
        X_enc,
        feature_names=X_sample.columns,
        show=False,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    print(f"✅ SHAP report saved ➜ {out_png}")


if __name__ == "__main__":
    main()
