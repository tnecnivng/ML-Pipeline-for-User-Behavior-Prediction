# scripts/make_new_users.py  (you can name it anything)

from __future__ import annotations

##############################################################################
# Add the *repository root* to sys.path so that `import ubp` always resolves #
##############################################################################
import sys, pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]   # “…/user-behavior-predictor”
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

##############################################################################
# Regular imports now work
##############################################################################
from ubp import data, features as ft
import pandas as pd
import pathlib as pl

# ────────────────────────────────────────────────────────────────────────────
# 1. load - clean the raw Online-Retail-II data
# ────────────────────────────────────────────────────────────────────────────
raw      = data.load_raw()
df_clean = data.clean_raw(raw)

# ────────────────────────────────────────────────────────────────────────────
# 2. build the 49-column feature table
# ────────────────────────────────────────────────────────────────────────────
X_full = ft.build_feature_table(df_clean)

# ────────────────────────────────────────────────────────────────────────────
# 3. drop target + date columns (the model doesn’t need them)
# ────────────────────────────────────────────────────────────────────────────
X = X_full.drop(
    columns=["did_repurchase_7d", "first_date", "last_date", "days_to_repurchase"]
)

# ────────────────────────────────────────────────────────────────────────────
# 4. take a sample (15 rows) to pretend they are “new” users
# ────────────────────────────────────────────────────────────────────────────
new_users = X.sample(15, random_state=0)

# ────────────────────────────────────────────────────────────────────────────
# 5. write it to disk
# ────────────────────────────────────────────────────────────────────────────
out = pl.Path("data") / "new_users.csv"
out.parent.mkdir(exist_ok=True)
new_users.to_csv(out, index=False)

print(f"✅  wrote {len(new_users)} rows to {out}")
