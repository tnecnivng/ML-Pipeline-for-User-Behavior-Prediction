from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pandas as pd

# ------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame, y) -> Pipeline:
    """Return an (unfitted) SMOTE-XGB pipeline with preprocessing."""
    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("bool").columns        # country dummies are bool

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", "passthrough", cat_cols)
    ])

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        learning_rate=0.05,
        n_estimators=400,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42,
    )

    pipe = Pipeline(steps=[
        ("pre",   pre),
        ("smote", SMOTETomek(random_state=42)),
        ("clf",   xgb)
    ])
    return pipe
