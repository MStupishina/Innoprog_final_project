import joblib
import json
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config
from src.telco_churn.classification.calibration import ProbabilityCalibrator
from src.telco_churn.classification.threshold import ThresholdTuner
from src.telco_churn.dataset.dataset import DatasetLoader
from src.telco_churn.dataset.encode_target import encode_target
from src.telco_churn.dataset.oof_generator import OOFPChurnGenerator
from src.telco_churn.model_factory import ModelFactory
from src.telco_churn.preprocessor import PreprocessorClassification
from src.telco_churn.trainer import ClassificationTrainer
from src.telco_churn.tuning import LGBMTuner, KNNTuner
from src.telco_churn.utils import make_json_serializable
from src.telco_churn.visualisation import plot_threshold_metrics, plot_roc_curve, plot_pr_curve, plot_confusion_matrix

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

    y_train_processed = encode_target(y_train, config.yes_no_map)
    y_val_processed = encode_target(y_val, config.yes_no_map)
    y_test_processed = encode_target(y_test, config.yes_no_map)

    model_path = config.model_dir / "churn_classification"
    model_path.mkdir(parents=True, exist_ok=True)

    best_artifacts = {
        "model_name": None,
        "pipeline": None,
        "calibrator": None,
        "threshold": 0.5,
        "score": -1,
        "params": None
    }

    for model_name in config.models_classification:
        print(f"\n=== Training model: {model_name} ===")
        model_threshold = 0.5
        calibrator = None
        best_params = None

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

        preprocessor = PreprocessorClassification(config)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        trainer = ClassificationTrainer(pipeline)
        trainer.fit(X_train, y_train_processed)

        # Калибровка и подбор порога только для LGBM и KNN
        if model_name.lower() in ["lightgbm", "knn"]:
            calibrator = ProbabilityCalibrator(method="sigmoid")

            X_val_cal, X_val_thresh, y_val_cal, y_val_thresh = train_test_split(
                X_val, y_val_processed,
                test_size=0.5, stratify=y_val_processed,
                random_state=config.random_state
            )

            calibrator.fit(pipeline, X_val_cal, y_val_cal)  # калибруем на одной половине
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

            test_proba = calibrator.predict_proba(X_test)
            test_pred = (test_proba >= model_threshold).astype(int)
        else:
            # Для LogisticRegression просто predict
            test_pred = trainer.predict(X_test)
            test_proba = trainer.predict_proba(X_test)[:, 1]

        print("\nValidation metrics:")
        val_metrics = trainer.evaluate(X_val, y_val_processed, model_threshold)
        for metric, value in val_metrics.items():
            print(f"{metric}: {value}")

        print("\nTest metrics:")
        test_metrics = trainer.evaluate(X_test, y_test_processed, model_threshold)
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")

        plot_roc_curve(
            y_true=y_test_processed,
            y_proba=test_proba,
            model_name=model_name.lower(),
            output_dir=model_path
        )

        plot_pr_curve(
            y_true=y_test_processed,
            y_proba=test_proba,
            model_name=model_name.lower(),
            output_dir=model_path
        )

        plot_confusion_matrix(
            y_true=y_test_processed,
            y_pred=test_pred,
            model_name=model_name.lower(),
            class_names=["No Churn", "Churn"],
            output_dir=model_path
        )

        current_score = val_metrics.get("pr_auc", 0)
        if current_score > best_artifacts["score"]:
            best_artifacts["score"] = current_score
            best_artifacts["model_name"] = model_name
            best_artifacts["pipeline"] = pipeline
            best_artifacts["calibrator"] = calibrator if model_name.lower() in ["lightgbm", "knn"] else None
            best_artifacts["threshold"] = model_threshold if model_name.lower() in ["lightgbm", "knn"] else 0.5
            best_artifacts["params"] = best_params

        metrics_to_save = {
            "validation": make_json_serializable(val_metrics),
            "test": make_json_serializable(test_metrics),
        }

        if model_name.lower() in ["lightgbm", "knn"]:
            metrics_to_save["best_threshold"] = float(model_threshold)

        # Сохраняем
        with open(model_path / f"{model_name}_metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=4)

        joblib.dump(pipeline, model_path / f"{model_name}_pipeline.joblib")
        if model_name.lower() in ["lightgbm", "knn"]:
            joblib.dump(calibrator, model_path / f"{model_name}_calibrator.joblib")
        print(f"\nArtifacts for {model_name} saved to models/")

    print(f"\n=== Best model: {best_artifacts['model_name']} with score {best_artifacts['score']:.4f} ===")

    best_model_name = best_artifacts["model_name"]
    best_calibrator = best_artifacts["calibrator"]
    best_threshold = best_artifacts["threshold"]
    best_params = best_artifacts["params"]

    # Сохраняем данные лучшей модели для инференса
    best_model_info = {
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "best_calibrator": best_calibrator,
        "best_params": best_params,
    }

    with open(model_path / "best_model_churn.json", "w") as f:
        json.dump(best_model_info, f, indent=4)

    final_model = ModelFactory.create_model(config=config, model_name=best_model_name, task_type="churn_classification")
    if best_params is not None:
        final_model.set_params(**best_params)

    final_preprocessor = PreprocessorClassification(config)
    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("model", final_model)
    ])

    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    y_full_encoded = encode_target(y_full, config.yes_no_map)

    final_trainer = ClassificationTrainer(final_pipeline)

    final_trainer.fit(X_full, y_full_encoded)

    if best_calibrator is not None:
        best_calibrator.fit(final_pipeline, X_val, y_val)

    # Считаем p_churn через OOF и сохраняем значения для регрессии ценности клиента
    train_val_df = pd.concat([train_df, val_df])
    oof_generator = OOFPChurnGenerator(config)
    oof_generator.generate_and_save(
        train_val_df, test_df,
        final_pipeline, best_calibrator
    )


if __name__ == "__main__":
    main()
