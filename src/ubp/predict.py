"""
Load the persisted pipeline and score a CSV of new user rows.
"""
import click, joblib, pandas as pd

@click.command()
@click.argument("csv_in", type=click.Path(exists=True))
@click.argument("csv_out", type=click.Path())
def main(csv_in, csv_out):
    pipe = joblib.load("models/xgb_smote_pipeline.pkl")
    X_new = pd.read_csv(csv_in)
    preds = pipe.predict_proba(X_new)[:, 1]
    pd.DataFrame({"prediction": preds}).to_csv(csv_out, index=False)
    print(f"✅  Saved predictions to {csv_out}")

if __name__ == "__main__":
    main()
