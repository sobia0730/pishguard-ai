"""Load the saved model and score URLs with the shared Phase 3 extractor."""

from pathlib import Path

import joblib
import pandas as pd

from src.features.preprocessing import FEATURE_COLUMNS, extract_url_features
from src.models.evaluation import LABEL_NAMES, SOURCE_TO_MODEL


MODEL_PATH = Path("models/phishguard_v1.joblib")


def load_model(path: str | Path = MODEL_PATH) -> dict:
    """Load a trusted local joblib artifact and verify its inference contract."""
    bundle = joblib.load(path)
    if bundle["feature_order"] != FEATURE_COLUMNS:
        raise ValueError("The artifact feature order differs from the URL extractor.")
    if bundle["source_to_model_label"] != SOURCE_TO_MODEL or bundle["label_names"] != LABEL_NAMES:
        raise ValueError("The artifact label mapping differs from the inference mapping.")
    return bundle


def phishing_probabilities(pipeline, features: pd.DataFrame):
    class_index = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(features)[:, class_index]


def predict_url(url: str, bundle: dict) -> dict:
    features = pd.DataFrame(
        [extract_url_features(url)], columns=bundle["feature_order"], dtype="int32"
    )
    probability = float(phishing_probabilities(bundle["pipeline"], features)[0])
    phishing = probability >= bundle["threshold"]
    return {
        "is_phishing": phishing,
        "prediction": bundle["label_names"][int(phishing)],
        "phishing_probability": probability,
        "model_version": bundle["model_version"],
    }
