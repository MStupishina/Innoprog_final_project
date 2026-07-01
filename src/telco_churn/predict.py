import pandas as pd

from src.telco_churn.inference.artifacts_loader import load_classification_artifacts, load_regression_artifacts
from src.telco_churn.inference.predictor import predict_all


def main():
    cls = load_classification_artifacts()
    reg = load_regression_artifacts()

    df = pd.read_csv("data/new_clients.csv")

    result = predict_all(
        churn_pipeline=cls["pipeline"],
        calibrator=cls["calibrator"],
        threshold=cls["threshold"],
        value_pipeline=reg["pipeline"],
        X=df
    )
    result.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()