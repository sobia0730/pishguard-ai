"""Tune models on Feast historical features and evaluate the winner once."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_store.materialize import ENTITY_FILE, REPO_PATH, SNAPSHOT_TIME, retrieve_historical
from src.features.preprocessing import FEATURE_COLUMNS, SEED
from src.models.evaluation import (
    LABEL_NAMES, MAX_VALIDATION_FPR, SOURCE_TO_MODEL, choose_threshold, encode_labels, evaluate,
)
from src.models.prediction import MODEL_PATH, phishing_probabilities


REPORT_PATH = Path("reports/model_evaluation.json")
MODEL_VERSION = "phishguard-v1"
GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "random_forest": [
        {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 4},
        {"n_estimators": 100, "max_depth": 18, "min_samples_leaf": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 2},
    ],
    "xgboost": [
        {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.10},
        {"n_estimators": 180, "max_depth": 4, "learning_rate": 0.10},
        {"n_estimators": 220, "max_depth": 5, "learning_rate": 0.08},
    ],
}
PACKAGES = ("feast", "joblib", "numpy", "pandas", "scikit-learn", "tldextract", "xgboost-cpu")


def build_pipeline(name: str, params: dict) -> Pipeline:
    if name == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, random_state=SEED, **params)),
        ])
    if name == "random_forest":
        model = RandomForestClassifier(random_state=SEED, n_jobs=2, **params)
    elif name == "xgboost":
        model = XGBClassifier(
            random_state=SEED, n_jobs=2, tree_method="hist", device="cpu",
            eval_metric="logloss", **params,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("model", model)])


def selection_key(trial: dict) -> tuple:
    metrics = trial["validation"]
    return (-metrics["recall"], metrics["false_positive_rate"], -metrics["pr_auc"])


def train_and_select(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[Pipeline, dict, dict, list]:
    """Fit on training data and select hyperparameters and threshold on validation."""
    x_train, x_validation = train[FEATURE_COLUMNS], validation[FEATURE_COLUMNS]
    y_train = encode_labels(train["label"])
    y_validation = encode_labels(validation["label"])
    best_models = {}
    comparison = {}
    trials = []
    for name, grid in GRIDS.items():
        for params in grid:
            pipeline = build_pipeline(name, params)
            start = perf_counter()
            pipeline.fit(x_train, y_train)
            duration = perf_counter() - start
            scores = phishing_probabilities(pipeline, x_validation)
            threshold = choose_threshold(y_validation, scores)
            trial = {
                "model": name,
                "params": params,
                "threshold": threshold,
                "fit_seconds": duration,
                "validation": evaluate(y_validation, scores, threshold),
            }
            trials.append(trial)
            if name not in comparison or selection_key(trial) < selection_key(comparison[name]):
                best_models[name] = pipeline
                comparison[name] = trial

    winner = min(comparison.values(), key=selection_key)
    return best_models[winner["model"]], winner, comparison, trials


def load_split(store: FeatureStore, split: str) -> pd.DataFrame:
    entity_path = REPO_PATH / "data" / ENTITY_FILE
    entities = pd.read_parquet(entity_path, filters=[("split", "==", split)])
    return retrieve_historical(store, entities, split)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        raise SystemExit(f"Final test was already recorded at {REPORT_PATH}.")
    store = FeatureStore(repo_path=str(REPO_PATH))
    train, validation = load_split(store, "train"), load_split(store, "validation")
    pipeline, winner, comparison, trials = train_and_select(train, validation)
    bundle = {
        "pipeline": pipeline,
        "model_version": MODEL_VERSION,
        "model_name": winner["model"],
        "params": winner["params"],
        "feature_order": FEATURE_COLUMNS.copy(),
        "source_to_model_label": SOURCE_TO_MODEL.copy(),
        "label_names": LABEL_NAMES.copy(),
        "threshold": winner["threshold"],
        "seed": SEED,
        "source_dataset": "LegitPhish v2",
        "feature_snapshot": SNAPSHOT_TIME.isoformat(),
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in PACKAGES},
    }

    # Reserve the final report before the one holdout lookup and evaluation.
    with REPORT_PATH.open("x", encoding="utf-8") as report_file:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH, compress=3)
        print(f"Validation selected {winner['model']} at threshold {winner['threshold']:.6f}", flush=True)

        test = load_split(store, "test")
        metrics = evaluate(
            encode_labels(test["label"]),
            phishing_probabilities(pipeline, test[FEATURE_COLUMNS]),
            winner["threshold"],
        )
        report = {
            "selected_model": {key: value for key, value in bundle.items() if key != "pipeline"},
            "artifact": MODEL_PATH.as_posix(),
            "row_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "selection": {
                "max_validation_false_positive_rate": MAX_VALIDATION_FPR,
                "ranking": ["highest phishing recall", "lowest false-positive rate", "highest PR-AUC"],
            },
            "confusion_matrix_labels": ["legitimate", "phishing"],
            "validation_comparison": comparison,
            "validation_trials": trials,
            "final_test": metrics,
        }
        json.dump(report, report_file, indent=2)
        report_file.write("\n")
    print(f"Final holdout evaluated: {len(test):,} rows. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
