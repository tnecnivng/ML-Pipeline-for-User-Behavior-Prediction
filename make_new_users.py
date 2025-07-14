
from __future__ import annotations

import sys, pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]   
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from ubp import data, features as ft
import pandas as pd
import pathlib as pl

raw      = data.load_raw()
df_clean = data.clean_raw(raw)

X_full = ft.build_feature_table(df_clean)

X = X_full.drop(
    columns=["did_repurchase_7d", "first_date", "last_date", "days_to_repurchase"]
)

new_users = X.sample(15, random_state=0)

out = pl.Path("data") / "new_users.csv"
out.parent.mkdir(exist_ok=True)
new_users.to_csv(out, index=False)

print(f"✅  wrote {len(new_users)} rows to {out}")
