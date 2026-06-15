import warnings

from configs.telco_churn_config import Config
from src.telco_churn.pipelines.classification_pipeline_steps import train_and_select_model, fit_final_pipeline, \
    save_artifacts
from src.telco_churn.datasets.dataset_loader import DatasetLoaderClassification
from src.telco_churn.datasets.encode_target import encode_target_classification
from src.telco_churn.datasets.oof_generator import OOFPChurnGenerator

# Подавляем предупреждения от LightGBM+Optuna
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna_integration.lightgbm"
)


def main():
    config = Config()
    dataset_loader = DatasetLoaderClassification(config)

    train_df, val_df, test_df, train_val_df = dataset_loader.load_and_split_data()
    dataset_loader.save_splits(train_df, val_df, test_df)

    X_train = train_df.drop(columns=[config.target_column_classification])
    y_train = encode_target_classification(train_df[config.target_column_classification], config.yes_no_map)

    X_val = val_df.drop(columns=[config.target_column_classification])
    y_val = encode_target_classification(val_df[config.target_column_classification], config.yes_no_map)

    X_test = test_df.drop(columns=[config.target_column_classification])
    y_test = encode_target_classification(test_df[config.target_column_classification], config.yes_no_map)

    X_train_val = train_val_df.drop(columns=[config.target_column_classification])
    y_train_val = encode_target_classification(train_val_df[config.target_column_classification], config.yes_no_map)

    artifact_path = config.artifacts_dir / "classification"
    artifact_path.mkdir(parents=True, exist_ok=True)

    best_result = train_and_select_model(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        selection_metric="roc_auc",
        artifact_path=artifact_path
    )

    print("\n=== BEST MODEL ===")
    print(f"Model: {best_result['model_name']}")
    print(f"Threshold: {best_result['threshold']:.3f}")
    print(f"Score: {best_result['score']:.4f}")

    final_result = fit_final_pipeline(
        config=config,
        best_result=best_result,
        X_train_val=X_train_val,
        y_train_val=y_train_val
    )

    save_artifacts(
        artifact_path=artifact_path,
        best_result=best_result,
        final_result=final_result,
        metrics=best_result["metrics"])

    # Считаем p_churn через OOF и сохраняем значения для регрессии ценности клиента
    oof_generator = OOFPChurnGenerator(config)
    oof_generator.generate_and_save(
        train_val_df=train_val_df,
        test_df=test_df,
        final_pipeline=final_result["pipeline"],
        final_calibrator=final_result["calibrator"]
    )


if __name__ == "__main__":
    main()
