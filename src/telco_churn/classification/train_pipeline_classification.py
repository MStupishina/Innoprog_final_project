import joblib
import json
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config
from src.telco_churn.classification.calibration import ProbabilityCalibrator
from src.telco_churn.classification.threshold import ThresholdTuner
from src.telco_churn.dataset import DatasetLoader
from src.telco_churn.model_factory import ModelFactory
from src.telco_churn.preprocessor import PreprocessorClassification
from src.telco_churn.trainer import ClassificationTrainer
from src.telco_churn.tuning import LGBMTuner, KNNTuner
from src.telco_churn.utils import make_json_serializable

# Подавляем предупреждения от LightGBM+Optuna
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna_integration.lightgbm"
)



def main():
    config = Config()
    dataset_loader = DatasetLoader(config)

    train_df, val_df, test_df = dataset_loader.load_and_split_data()
    dataset_loader.save_splits(train_df, val_df, test_df)

    X_train = train_df.drop(columns=[config.target_column])
    y_train = train_df[config.target_column]

    X_val = val_df.drop(columns=[config.target_column])
    y_val = val_df[config.target_column]

    X_test = test_df.drop(columns=[config.target_column])
    y_test = test_df[config.target_column]

    preprocessor = PreprocessorClassification(config)
    y_train_processed = preprocessor.transform_target(y_train)
    y_val_processed = preprocessor.transform_target(y_val)
    y_test_processed = preprocessor.transform_target(y_test)

    model_path = config.model_dir / "churn_classification"
    model_path.mkdir(exist_ok=True)

    best_model_name = None
    best_metric = -1

    # Сохранение лучшего pipeline
    best_pipeline = None
    best_calibrator = None
    best_model_threshold = None
    calibrator = None

    model_threshold = 0.5

    for model_name in config.models_classification:
        print(f"\n=== Training model: {model_name} ===")
        model = ModelFactory.create_model(config, model_name, task_type="churn_classification")

        # Если модель не baseline, выполняем tuning
        if model_name.lower() == "lightgbm":
            tuner = LGBMTuner(config)
            best_params = tuner.tune(X_train, y_train_processed, X_val, y_val_processed)
            model.set_params(**best_params)
        elif model_name.lower() == "knn":
            tuner = KNNTuner(config)
            best_params = tuner.tune(X_train, y_train_processed, X_val, y_val_processed)
            model.set_params(**best_params)

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        trainer = ClassificationTrainer(model, preprocessor)
        trainer.fit(pipeline)

        # Калибровка и подбор порога только для LGBM и KNN
        if model_name.lower() in ["lightgbm", "knn"]:
            calibrator = ProbabilityCalibrator(method="sigmoid")

            X_val_cal, X_val_thresh, y_val_cal, y_val_thresh = train_test_split(
                X_val, y_val_processed,
                test_size=0.5, random_state=config.random_state
            )

            calibrator.fit(trainer.model, X_val_cal, y_val_cal)  # калибруем на одной половине
            val_proba = calibrator.predict_proba(X_val_thresh)  # порог выбираем на другой
            threshold_tuner = ThresholdTuner()
            model_threshold, best_score = threshold_tuner.tune(
                y_val_thresh, val_proba
            )

            print("\nBest threshold:", model_threshold)
            print("Best F1:", best_score)
            plot_threshold_metrics(
                y_val_thresh,
                val_proba,
                model_name=model_name.lower(),
                output_dir=model_path  # сохраняем в папку с моделями
            )

            val_proba_full = calibrator.predict_proba(X_val)
            val_pred_full = (val_proba_full>=model_threshold).astype(int)
            test_proba = calibrator.predict_proba(X_test)
            test_pred = (test_proba >= model_threshold).astype(int)
        else:
            # Для LogisticRegression просто predict
            val_pred_full = trainer.model.predict(X_val)
            val_proba_full = trainer.model.predict_proba(X_val)[:, 1]
            test_pred = trainer.model.predict(X_test)
            test_proba = trainer.model.predict_proba(X_test)[:, 1]

        print("\nValidation metrics:")
        val_metrics = trainer.evaluate(y_val_processed, val_pred_full, val_proba_full)
        for metric, value in val_metrics.items():
            print(f"{metric}: {value}")

        print("\nTest metrics:")
        test_metrics = trainer.evaluate(y_test_processed, test_pred, test_proba)
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")

        current_score = val_metrics.get("pr_auc", 0)
        if current_score > best_metric:
            best_metric = current_score
            best_model_name = model_name
            best_model = trainer.model
            best_preprocessor = preprocessor
            best_calibrator = calibrator if model_name.lower() in ["lightgbm", "knn"] else None
            best_model_threshold = model_threshold if model_name.lower() in ["lightgbm", "knn"] else 0.5

        metrics_to_save = {
            "validation": make_json_serializable(val_metrics),
            "test": make_json_serializable(test_metrics),
        }

        if model_name.lower() in ["lightgbm", "knn"]:
            metrics_to_save["best_threshold"] = float(best_model_threshold)

        # Сохраняем
        with open(model_path / f"{model_name}_metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=4)

        joblib.dump(trainer.model, model_path / f"{model_name}_model.joblib")
        joblib.dump(trainer.preprocessor, model_path / f"{model_name}_preprocessor.joblib")
        if model_name.lower() in ["lightgbm", "knn"]:
            joblib.dump(calibrator, model_path / f"{model_name}_calibrator.joblib")
        print(f"\nArtifacts for {model_name} saved to models/")

    print(f"\n=== Best model: {best_model_name} with score {best_metric:.4f} ===")

    # Сохраняем имя лучшей модели для инференса
    best_model_info = {"best_model_name": best_model_name}
    with open(model_path / "best_model_churn.json", "w") as f:
        json.dump(best_model_info, f)

    #Считаем p_churn через OOF и сохраняем значения для регрессии ценности клиента
    train_val_df = pd.concat([train_df, val_df])
    oof_generator = OOFPChurnGenerator(config)
    oof_generator.generate_and_save(
        train_val_df, test_df,
        best_model_name, best_model,
        best_preprocessor, best_calibrator
    )


if __name__ == "__main__":
    main()