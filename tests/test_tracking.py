import json
from importlib.metadata import version

import joblib
import mlflow
import pandas as pd

from src.features.preprocessing import FEATURE_COLUMNS, extract_url_features
from src.models.evaluation import LABEL_NAMES, SOURCE_TO_MODEL
from src.models.prediction import predict_url
from src.models.tracking import (
    CANDIDATE_ALIAS, EXPERIMENT_NAME, MODEL_NAME, RETRAINING_ALIAS, SELECTION_STATUS, import_results,
)
from src.models.train import PACKAGES, build_pipeline


def test_tracks_trials_registers_winner_and_reuses_completed_batch(tmp_path):
    prior_tracking = mlflow.get_tracking_uri()
    prior_registry = mlflow.get_registry_uri()
    url = "https://example.test/login"
    sample = pd.DataFrame(
        [extract_url_features(url), extract_url_features("https://example.test/")],
        columns=FEATURE_COLUMNS,
        dtype="int32",
    )
    params = {"C": 1.0}
    pipeline = build_pipeline("logistic_regression", params).fit(sample, [1, 0])
    bundle = {
        "pipeline": pipeline,
        "model_name": "logistic_regression",
        "model_version": "fixture",
        "params": params,
        "threshold": 0.5,
        "feature_order": FEATURE_COLUMNS,
        "seed": 42,
        "source_to_model_label": SOURCE_TO_MODEL,
        "label_names": LABEL_NAMES,
        "package_versions": {name: version(name) for name in PACKAGES},
    }
    metrics = {"recall": 1.0, "confusion_matrix": [[1, 0], [0, 1]]}
    winner = {"model": bundle["model_name"], "params": params, "threshold": 0.5,
              "fit_seconds": 0.1, "validation": metrics}
    other = {**winner, "model": "random_forest"}
    report = {
        "selected_model": {key: value for key, value in bundle.items() if key != "pipeline"},
        "validation_comparison": {bundle["model_name"]: winner},
        "validation_trials": [other, winner],
        "final_test": metrics,
    }
    report_path, bundle_path = tmp_path / "evaluation.json", tmp_path / "model.joblib"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    joblib.dump(bundle, bundle_path)
    provenance = {"dataset_sha256": "fixture", "git_commit_at_import": "unavailable"}

    try:
        first = import_results(report_path, bundle_path, tmp_path / "mlflow", provenance)
        assert first == import_results(report_path, bundle_path, tmp_path / "mlflow", provenance)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 2
        selected = client.get_run(first["run_id"])
        assert selected.data.metrics["test_recall"] == 1.0
        assert sum("test_recall" in run.data.metrics for run in runs) == 1
        assert {item.path for item in client.list_artifacts(first["run_id"], "evaluation")} >= {
            "evaluation/evaluation.json", "evaluation/provenance.json",
            "evaluation/test_confusion_matrix.json", "evaluation/validation_confusion_matrix.json",
        }
        registered = client.get_model_version_by_alias(MODEL_NAME, CANDIDATE_ALIAS)
        assert str(registered.version) == first["version"]
        assert registered.tags["approval_status"] == SELECTION_STATUS
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CANDIDATE_ALIAS}")
        assert model.predict(pd.DataFrame({"url": [url]})).iloc[0].to_dict() == predict_url(url, bundle)

        next_report = {key: value for key, value in report.items() if key != "final_test"}
        next_report["selected_model"] = {**next_report["selected_model"], "model_version": "fixture-candidate"}
        report_path.write_text(json.dumps(next_report), encoding="utf-8")
        joblib.dump({**bundle, "model_version": "fixture-candidate"}, bundle_path)
        pending = import_results(
            report_path, bundle_path, tmp_path / "mlflow",
            {**provenance, "dvc_raw_md5": "dvc-fixture", "training_git_commit": "git-fixture"},
            alias=RETRAINING_ALIAS,
        )
        future_run = client.get_run(pending["run_id"])
        assert "test_recall" not in future_run.data.metrics
        assert future_run.data.tags["dvc_raw_md5"] == "dvc-fixture"
        assert future_run.data.tags["training_git_commit"] == "git-fixture"
        assert str(client.get_model_version_by_alias(MODEL_NAME, CANDIDATE_ALIAS).version) == first["version"]
        assert str(client.get_model_version_by_alias(MODEL_NAME, RETRAINING_ALIAS).version) == pending["version"]
    finally:
        mlflow.set_tracking_uri(prior_tracking)
        mlflow.set_registry_uri(prior_registry)
