import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import FEATURE_COLUMNS, extract_url_features
from src.models.evaluation import LABEL_NAMES, SOURCE_TO_MODEL, choose_threshold, encode_labels, evaluate
from src.models.prediction import load_model, phishing_probabilities, predict_url
from src.models.train import train_and_select


@pytest.fixture(scope="module")
def miniature_training():
    def make_rows(start):
        rows = []
        for i in range(start, start + 24):
            for url, label in (
                (f"http://login{i}.example.test/verify-account", 0),
                (f"https://www.example.test/about/{i}", 1),
            ):
                rows.append({**extract_url_features(url), "label": label})
        return pd.DataFrame(rows)

    return train_and_select(make_rows(0), make_rows(30))


def test_source_label_conversion():
    assert encode_labels(pd.Series([0, 1, 0])).tolist() == [1, 0, 1]
    with pytest.raises(ValueError, match="original labels"):
        encode_labels(pd.Series([0, 2]))


def test_metrics_use_phishing_positive_and_confusion_order():
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate(labels, np.array([0.2, 0.8, 0.7, 0.9]), 0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate"):
        assert metrics[key] == 0.5
    assert 0.5 < metrics["pr_auc"] < 1


def test_threshold_prioritizes_recall_within_cap():
    labels = np.array([0] * 10 + [1, 1, 1])
    scores = np.array([0.8, 0.6, *([0.1] * 8), 0.9, 0.7, 0.5])
    cutoff = choose_threshold(labels, scores)
    assert cutoff == 0.7
    metrics = evaluate(labels, scores, cutoff)
    assert metrics["false_positive_rate"] == 0.1
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_trains_and_compares_three_model_families(miniature_training):
    pipeline, winner, comparison, trials = miniature_training
    assert set(comparison) == {"logistic_regression", "random_forest", "xgboost"}
    assert len(trials) == 9
    assert all(trial["validation"]["false_positive_rate"] <= 0.1 for trial in trials)
    assert winner == comparison[winner["model"]]
    assert list(pipeline.classes_) == [0, 1]


def test_model_roundtrip_and_shared_prediction_features(tmp_path, miniature_training):
    pipeline, winner, _, _ = miniature_training
    path = tmp_path / "bundle.joblib"
    bundle = {
        "pipeline": pipeline,
        "model_version": "unit-test",
        "feature_order": FEATURE_COLUMNS,
        "source_to_model_label": SOURCE_TO_MODEL,
        "label_names": LABEL_NAMES,
        "threshold": winner["threshold"],
    }
    joblib.dump(bundle, path)
    loaded = load_model(path)
    url = "http://login70.example.test/verify-account"
    frame = pd.DataFrame([extract_url_features(url)], columns=FEATURE_COLUMNS, dtype="int32")
    result = predict_url(url, loaded)
    assert result["phishing_probability"] == pytest.approx(phishing_probabilities(pipeline, frame)[0])
    assert result["is_phishing"]
    assert result["prediction"] == "phishing"
    assert result["model_version"] == "unit-test"

    joblib.dump({**bundle, "feature_order": FEATURE_COLUMNS[::-1]}, path)
    with pytest.raises(ValueError, match="feature order"):
        load_model(path)
