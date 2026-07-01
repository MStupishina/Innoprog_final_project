import pandas as pd

from configs.telco_churn_config import Config


def predict_churn(
        pipeline,
        calibrator,
        threshold: float,
        X: pd.DataFrame
) -> pd.DataFrame:
    if calibrator is not None:
        proba = calibrator.predict_proba(X)
    else:
        proba = pipeline.predict_proba(X)

    pred = (proba >= threshold).astype(int)
    return pd.DataFrame({
        "churn_probability": proba,
        "churn_prediction": pred
    })


def predict_value(
        pipeline,
        X: pd.DataFrame
) -> pd.Series:
    return pd.Series(
        pipeline.predict(X),
        name="predicted_value"
    )


def predict_all(
        churn_pipeline,
        calibrator,
        threshold,
        value_pipeline,
        X: pd.DataFrame
) -> pd.DataFrame:
    churn = predict_churn(churn_pipeline, calibrator, threshold, X)
    value = predict_value(value_pipeline, X)
    return pd.concat([churn, value], axis=1)




