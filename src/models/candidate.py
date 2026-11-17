"""Train a future validation-only candidate. The Phase 5 holdout stays closed."""

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import yaml

from feature_store.materialize import SNAPSHOT_TIME, register_features
from src.data.ingestion import DEFAULT_CSV
from src.features.preprocessing import FEATURE_COLUMNS, SEED
from src.models.evaluation import LABEL_NAMES, MAX_VALIDATION_FPR, SOURCE_TO_MODEL
from src.models.train import PACKAGES, load_split, train_and_select


CANDIDATE_MODEL = Path("models/candidates/candidate.joblib")
CANDIDATE_REPORT = Path("reports/candidate_validation.json")


def file_hash(path: Path, algorithm: str) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def candidate_provenance() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    marker = DEFAULT_CSV.with_suffix(".csv.dvc")
    raw = yaml.safe_load(marker.read_text(encoding="utf-8"))["outs"][0]["md5"]
    if raw != file_hash(DEFAULT_CSV, "md5"):
        raise ValueError("Raw CSV differs from its DVC pointer.")
    locked = yaml.safe_load(Path("dvc.lock").read_text(encoding="utf-8"))["stages"]["preprocess"]["outs"]
    hashes = {item["path"]: item["md5"] for item in locked}
    splits = {}
    for name in ("train", "validation"):
        path = Path(f"data/processed/{name}.csv")
        expected = hashes[path.as_posix()]
        if expected != file_hash(path, "md5"):
            raise ValueError(f"{name} split differs from its DVC lock entry.")
        splits[name] = expected
    return {
        "dataset": "LegitPhish v2",
        "dataset_sha256": file_hash(DEFAULT_CSV, "sha256"),
        "dvc_raw_md5": raw,
        "dvc_training_split_md5": splits,
        "training_git_commit": commit,
        "git_worktree_dirty": bool(dirty.strip()),
    }


def train_validation_candidate(store, provenance: dict) -> tuple[dict, dict]:
    train = load_split(store, "train")
    validation = load_split(store, "validation")
    pipeline, winner, comparison, trials = train_and_select(train, validation)
    identifier = hashlib.sha256(json.dumps({
        "provenance": provenance,
        "model": winner["model"],
        "params": winner["params"],
        "threshold": winner["threshold"],
    }, sort_keys=True).encode()).hexdigest()[:12]
    bundle = {
        "pipeline": pipeline,
        "model_version": f"phishguard-candidate-{identifier}",
        "model_name": winner["model"],
        "params": winner["params"],
        "feature_order": FEATURE_COLUMNS.copy(),
        "source_to_model_label": SOURCE_TO_MODEL.copy(),
        "label_names": LABEL_NAMES.copy(),
        "threshold": winner["threshold"],
        "seed": SEED,
        "source_dataset": provenance["dataset"],
        "feature_snapshot": SNAPSHOT_TIME.isoformat(),
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in PACKAGES},
    }
    report = {
        "selected_model": {key: value for key, value in bundle.items() if key != "pipeline"},
        "row_counts": {"train": len(train), "validation": len(validation)},
        "selection": {"max_validation_false_positive_rate": MAX_VALIDATION_FPR},
        "validation_comparison": comparison,
        "validation_trials": trials,
        "provenance": provenance,
    }
    return bundle, report


def main() -> None:
    provenance = candidate_provenance()
    bundle, report = train_validation_candidate(register_features(), provenance)
    CANDIDATE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, CANDIDATE_MODEL, compress=3)
    CANDIDATE_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation-only candidate: {bundle['model_version']}")


if __name__ == "__main__":
    main()
