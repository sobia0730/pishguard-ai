"""MLflow's URL prediction adapter for the existing PhishGuard model bundle."""

import mlflow
import pandas as pd

from src.models.prediction import load_model, predict_url


class PhishGuardModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = load_model(context.artifacts["bundle"])

    def predict(self, context, model_input):
        return pd.DataFrame([predict_url(url, self.bundle) for url in model_input["url"]])


mlflow.models.set_model(PhishGuardModel())
