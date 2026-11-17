from pathlib import Path
from unittest.mock import Mock

import yaml

from src.models import candidate


def test_pipeline_tracks_expected_outputs_and_excludes_holdout_scoring():
    root = Path(__file__).resolve().parents[1]
    stages = yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    pointer = yaml.safe_load((root / "data/raw/url_features_extracted1.csv.dvc").read_text())
    assert pointer["outs"][0]["path"] == "url_features_extracted1.csv"
    assert len(pointer["outs"][0]["md5"]) == 32
    assert set(stages) == {"validate", "preprocess", "prepare_feast", "train_candidate"}
    expected = {f"data/processed/{name}.csv" for name in ("train", "validation", "test")}
    assert expected <= set(stages["preprocess"]["outs"])
    assert expected <= set(stages["prepare_feast"]["deps"])
    assert set(stages["prepare_feast"]["outs"]) <= set(stages["train_candidate"]["deps"])
    assert stages["train_candidate"]["cmd"] == "python -m src.models.candidate"
    outputs = {out for stage in stages.values() for out in stage["outs"]}
    assert not outputs & {"models/phishguard_v1.joblib", "reports/model_evaluation.json",
                          "mlflow_local/tracking.db"}


def test_future_candidate_only_loads_train_and_validation(monkeypatch):
    loader = Mock(side_effect=[[1, 2], [3]])
    monkeypatch.setattr(candidate, "load_split", loader)
    winner = {"model": "fixture", "params": {}, "threshold": 0.5}
    monkeypatch.setattr(
        candidate, "train_and_select", lambda train, val: ("pipeline", winner, {"fixture": winner}, [winner])
    )
    provenance = {"dataset": "LegitPhish v2", "dvc_raw_md5": "fixture", "training_git_commit": "fixture"}
    bundle, report = candidate.train_validation_candidate("store", provenance)
    assert [call.args[1] for call in loader.call_args_list] == ["train", "validation"]
    assert "final_test" not in report
    assert report["provenance"] == provenance
    assert bundle["model_version"].startswith("phishguard-candidate-")
