import pandas as pd

from configs.telco_churn_config import Config
from src.telco_churn.inference.artifacts_loader import load_classification_artifacts, load_regression_artifacts
from src.telco_churn.inference.predictor import predict_all


def main():
    config = Config()
    cls = load_classification_artifacts()
    reg = load_regression_artifacts()

    df = pd.read_csv(config.inference_input)

    result = predict_all(
        churn_pipeline=cls["pipeline"],
        calibrator=cls["calibrator"],
        threshold=cls["threshold"],
        value_pipeline=reg["pipeline"],
        X=df
    )
    config.inference_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.inference_output, index=False)
    print(f"Predictions saved to {config.inference_output}")

if __name__ == "__main__":
    main()