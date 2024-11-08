import joblib
import os


class RFModel:
    def __init__(
        self, model_path=os.path.join(os.path.dirname(__file__), "ml_model.pkl")
    ):
        self.model = joblib.load(model_path)

    def load_model(self):
        return self.model
