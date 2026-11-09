"""Recreates the midterm pipeline and saves it with joblib.
REPLACE the data/feature sections with your ACTUAL midterm notebook code so results match.
Run once locally:  python train_model.py   (the app never retrains on refresh)
"""
import datetime as dt
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from modules.data_processing import CATEGORY_OPTIONS, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET

SEED = 42


def make_synthetic(n=3000, seed=SEED):
    """Synthetic, non-sensitive data. Swap for your midterm dataset (e.g., pd.read_csv)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "request_id": [f"REQ-{i:05d}" for i in range(n)],
        "request_type": rng.choice(CATEGORY_OPTIONS["request_type"], n),
        "department": rng.choice(CATEGORY_OPTIONS["department"], n),
        "channel": rng.choice(CATEGORY_OPTIONS["channel"], n, p=[0.4, 0.3, 0.3]),
        "priority": rng.choice(CATEGORY_OPTIONS["priority"], n, p=[0.2, 0.6, 0.2]),
        "queue_load": rng.integers(5, 120, n),
        "staff_on_duty": rng.integers(1, 9, n),
        "attachments": rng.integers(0, 6, n),
        "submitted_hour": rng.integers(7, 18, n),
        "is_peak_period": rng.choice([0, 1], n, p=[0.75, 0.25]),
    })
    z = (-2.2 + 0.03 * df.queue_load - 0.35 * df.staff_on_duty + 0.9 * df.is_peak_period
         + 0.25 * df.attachments + 0.8 * df.request_type.isin(["Records/TOR", "Clearance"])
         + 0.4 * (df.channel == "Walk-in") - 0.5 * (df.priority == "High")
         + 0.5 * (df.submitted_hour >= 16) + rng.normal(0, 0.6, n))
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def main():
    df = make_synthetic()
    df.to_csv("data/sample_requests.csv", index=False)
    X, y = df[FEATURE_COLUMNS], df[TARGET]   # leakage columns (e.g., resolution_days) excluded
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORY_OPTIONS)),
        ("num", StandardScaler(), NUMERIC_FEATURES)])
    pipe = Pipeline([("prep", pre),
                     ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                      class_weight="balanced", random_state=SEED))])
    pipe.fit(X_tr, y_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    metrics = {"Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
               "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
               "ROC-AUC": roc_auc_score(y_te, proba)}
    joblib.dump({"pipeline": pipe, "features": FEATURE_COLUMNS, "metrics": metrics,
                 "name": "RandomForest SLA-breach classifier", "version": "1.0",
                 "trained_at": dt.date.today().isoformat()}, "models/trained_model.joblib")
    X_te.assign(**{TARGET: y_te}).head(5).to_csv("data/test_cases.csv", index=False)  # for consistency check
    print({k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
