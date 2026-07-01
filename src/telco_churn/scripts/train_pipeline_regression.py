import warnings

from configs.telco_churn_config import Config
from src.telco_churn.datasets.dataset_loader import DatasetLoaderRegression
from src.telco_churn.pipelines.regression_pipeline_steps import train_and_select_model, save_artifacts

# Подавляем предупреждения от LightGBM+Optuna
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna_integration.lightgbm"
)

def main():
    config = Config()

    loader = DatasetLoaderRegression(config)
    train_df, test_df = loader.load_and_split_data()

    X_train = train_df.drop(columns=[config.target_column_value, "p_churn"])
    y_train = train_df[config.target_column_value]

    X_test = test_df.drop(columns=[config.target_column_value, "p_churn"])
    y_test = test_df[config.target_column_value]

    artifacts_path = config.artifacts_dir / "value_regression"
    artifacts_path.mkdir(parents=True, exist_ok=True)

    best_results = train_and_select_model(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        artifact_path=artifacts_path,
    )

    save_artifacts(
        artifact_path=artifacts_path,
        best_results=best_results
    )

if __name__ == "__main__":
    main()