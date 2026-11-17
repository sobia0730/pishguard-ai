"""Import completed PhishGuard trials and register the selected URL model."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from src.data.ingestion import DEFAULT_CSV
from src.features.preprocessing import FEATURE_REPORT, OUTPUT_DIR
from src.models.prediction import MODEL_PATH, load_model
from src.models.train import REPORT_PATH


ROOT = Path(__file__).resolve().parents[2]
TRACKING_DIR = ROOT / "mlflow_local"
EXPERIMENT_NAME = "PhishGuard AI"
MODEL_NAME = "PhishGuardURLClassifier"
CANDIDATE_ALIAS = "candidate"
RETRAINING_ALIAS = "retraining-candidate"
SELECTION_STATUS = "candidate_pending_approval"


def tracking_uri(directory: Path) -> str:
    return f"sqlite:///{(directory.resolve() / 'tracking.db').as_posix()}"


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def log_evaluation(prefix: str, metrics: dict) -> None:
    scores = {f"{prefix}_{name}": value for name, value in metrics.items() if name != "confusion_matrix"}
    mlflow.log_metrics(scores)
    mlflow.log_dict(
        {"labels": ["legitimate", "phishing"], "matrix": metrics["confusion_matrix"]},
        f"evaluation/{prefix}_confusion_matrix.json",
    )


def verify_bundle(report: dict, bundle: dict) -> None:
    metadata = report["selected_model"]
    for key in ("model_name", "model_version", "params", "threshold", "feature_order", "seed"):
        if metadata[key] != bundle[key]:
            raise ValueError(f"Saved report and model bundle disagree on {key}.")
    comparison = report["validation_comparison"]
    if not any(trial == comparison[bundle["model_name"]] for trial in report["validation_trials"]):
        raise ValueError("The selected validation trial is missing.")


def register_selected_model(bundle_path: Path, bundle: dict):
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "url")]),
        outputs=Schema([
            ColSpec("boolean", "is_phishing"),
            ColSpec("string", "prediction"),
            ColSpec("double", "phishing_probability"),
            ColSpec("string", "model_version"),
        ]),
    )
    needed = ("joblib", "numpy", "pandas", "scikit-learn", "tldextract", "xgboost-cpu")
    requirements = [f"mlflow=={mlflow.__version__}"]
    requirements.extend(f"{name}=={bundle['package_versions'][name]}" for name in needed)
    return mlflow.pyfunc.log_model(
        name="url_classifier",
        python_model=str(ROOT / "src" / "models" / "registry_model.py"),
        code_paths=[str(ROOT / "src")],
        artifacts={"bundle": str(bundle_path.resolve())},
        signature=signature,
        input_example=pd.DataFrame({"url": ["https://example.test/account"]}),
        pip_requirements=requirements,
        registered_model_name=MODEL_NAME,
        metadata={"model_version": bundle["model_version"]},
    )


def import_results(
    report_path: Path, bundle_path: Path, directory: Path, provenance: dict,
    *, alias: str = CANDIDATE_ALIAS,
) -> dict:
    """Log all saved trials once and return the winner run and registry version."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bundle = load_model(bundle_path)
    verify_bundle(report, bundle)
    batch_id = hashlib.sha256(
        (sha256_file(report_path) + sha256_file(bundle_path)).encode("ascii")
    ).hexdigest()

    directory.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri(directory)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        EXPERIMENT_NAME, artifact_location=(directory.resolve() / "artifacts").as_uri()
    )
    completed = {
        run.data.tags["trial_key"]: run
        for run in client.search_runs(
            [experiment_id], f"tags.batch_id = '{batch_id}' AND attributes.status = 'FINISHED'"
        )
    }
    selected_trial = report["validation_comparison"][bundle["model_name"]]
    winner_run = None
    registry_version = None
    for index, trial in enumerate(report["validation_trials"], start=1):
        key = str(index)
        selected = trial == selected_trial
        if key in completed:
            if selected:
                winner_run = completed[key].info.run_id
                registry_version = completed[key].data.tags["registry_version"]
            continue

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{trial['model']}-{index:02d}",
            tags={
                "batch_id": batch_id, "trial_key": key,
                "dataset_version": provenance["dataset_sha256"],
                "git_commit": provenance.get("training_git_commit", provenance.get("git_commit_at_import", "unavailable")),
                "training_git_commit": provenance.get("training_git_commit", "unavailable"),
                "dvc_raw_md5": provenance.get("dvc_raw_md5", "unavailable"),
                "approval_status": SELECTION_STATUS if selected else "not_selected",
                "tracking_origin": "phase5_saved_report" if alias == CANDIDATE_ALIAS else "dvc_validation_only",
            },
        ) as run:
            mlflow.log_params({
                "model": trial["model"], "seed": bundle["seed"], "threshold": trial["threshold"],
                **trial["params"],
            })
            mlflow.log_metric("fit_seconds", trial["fit_seconds"])
            log_evaluation("validation", trial["validation"])
            mlflow.log_dict(provenance, "evaluation/provenance.json")
            if selected:
                if "final_test" in report:
                    log_evaluation("test", report["final_test"])
                mlflow.log_artifact(str(report_path), "evaluation")
                info = register_selected_model(bundle_path, bundle)
                registry_version = str(info.registered_model_version)
                winner_run = run.info.run_id
                client.set_model_version_tag(MODEL_NAME, registry_version, "approval_status", SELECTION_STATUS)
                mlflow.set_tag("registry_version", registry_version)

    if not winner_run or not registry_version:
        raise RuntimeError("The import did not locate a registered winner.")
    client.set_registered_model_alias(MODEL_NAME, alias, registry_version)
    return {"run_id": winner_run, "version": registry_version, "trials": len(report["validation_trials"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true", help="Import future DVC validation-only results.")
    args = parser.parse_args()
    if args.candidate:
        from src.models.candidate import CANDIDATE_MODEL, CANDIDATE_REPORT

        report = json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8"))
        if "final_test" in report:
            raise ValueError("A validation-only candidate must not contain final test metrics.")
        result = import_results(
            CANDIDATE_REPORT, CANDIDATE_MODEL, TRACKING_DIR, report["provenance"],
            alias=RETRAINING_ALIAS,
        )
        print(f"Tracked {result['trials']} validation trials.")
        print(f"Registered models:/{MODEL_NAME}/{result['version']} with alias @{RETRAINING_ALIAS}.")
        return

    provenance = {
        "dataset": "LegitPhish v2",
        "dataset_sha256": sha256_file(DEFAULT_CSV),
        "split_sha256_at_import": {
            split: sha256_file(OUTPUT_DIR / f"{split}.csv")
            for split in ("train", "validation", "test")
        },
        "feature_report_sha256_at_import": sha256_file(FEATURE_REPORT),
        "git_commit_at_import": git_commit(),
        "training_git_commit": "unavailable",
    }
    result = import_results(REPORT_PATH, MODEL_PATH, TRACKING_DIR, provenance)
    print(f"Tracked {result['trials']} trials in {EXPERIMENT_NAME}.")
    print(f"Selected run: {result['run_id']}")
    print(f"Registered models:/{MODEL_NAME}/{result['version']} with alias @{CANDIDATE_ALIAS}.")


if __name__ == "__main__":
    main()
