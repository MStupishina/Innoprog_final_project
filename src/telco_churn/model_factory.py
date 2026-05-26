from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPRegressor

from configs.Telco_churn_config import Config


class ModelFactory:
    """Фабрика для создания ML-моделей для задач классификации churn и regression value"""
    @staticmethod
    def create_model(config: Config, model_name: str, task_type: str):
        if not model_name:
            raise ValueError("Имя модели не может быть пустым")

        if not task_type:
            raise ValueError("Тип задачи не может быть пустым")

        model_name = model_name.lower()
        task_type = task_type.lower()

        valid_tasks = ["churn_classification", "value_regression"]

        if task_type not in valid_tasks:
            raise ValueError(
                f"Тип задачи не поддерживается: {task_type}. Поддерживаются: {valid_tasks}"
            )

        # Churn classification
        if task_type == "churn_classification":
            if model_name == "logistic_regression":
                return LogisticRegression(
                    C=config.lr_C,
                    max_iter=config.lr_max_iter,
                    solver=config.lr_solver,
                    random_state=config.random_state,
                )

            elif model_name == "lightgbm":

                return LGBMClassifier(
                    n_estimators=config.lgbm_n_estimators,
                    learning_rate=config.lgbm_learning_rate,
                    num_leaves=config.lgbm_num_leaves,
                    random_state=config.random_state,
                )
            elif model_name == "knn":
                return KNeighborsClassifier(n_neighbors=config.knn_n_neighbors)

            else:
                raise ValueError(f"Неподдерживаемая модель: {model_name} для задачи {task_type}")

        # Value regression
        if task_type == "value_regression":

            if model_name == "ridge":
                return Ridge(alpha=config.ridge_alpha)

            elif model_name == "lightgbm":
                return LGBMRegressor(
                    n_estimators=config.lgbm_r_n_estimators,
                    learning_rate=config.lgbm_r_learning_rate,
                    num_leaves=config.lgbm_r_num_leaves,
                    random_state=config.random_state,
                )

            elif model_name == "mlp":
                return MLPRegressor(
                    hidden_layer_sizes=config.mlp_hidden_layer_sizes,
                    max_iter=config.mlp_max_iter,
                    random_state=config.random_state,
                )
            else:
                raise ValueError(f"Неподдерживаемая модель: {model_name} для задачи {task_type}")
        raise ValueError(f"Неподдерживаемый тип задачи: {task_type}")