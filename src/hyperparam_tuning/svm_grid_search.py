import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "SVM"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def grid_search_svm(max_samples=10000):
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, normalize=True)

        # Ограничиваем выборку для ускорения подбора параметров
        X_train_small = X_train[:max_samples]
        y_train_small = y_train[:max_samples]

        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]  # Влияет только на rbf kernel
        }

        clf = svm.SVC()

        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2
        )

        results_logger.info(f"Starting GridSearchCV for {MODEL_NAME} on {max_samples} samples...")

        grid_search.fit(X_train_small, y_train_small)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        results_logger.info(f"GridSearchCV finished.")
        results_logger.info(f"Best params: {best_params}, Best accuracy: {best_score:.4f}")
        print(f"Best params: {best_params}, Best accuracy: {best_score:.4f}")

        # Логируем только гиперпараметры и accuracy (без start/stop для метрик)
        metric_logger.set_hyperparameters(best_params, best_score)
        metric_logger.log_all()

        return best_params, best_score

    except Exception as e:
        error_logger.error(f"Error during GridSearchCV for {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при подборе гиперпараметров: {e}")
        return None, None

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} GridSearch START ===")
    grid_search_svm()
    results_logger.info(f"=== {MODEL_NAME} GridSearch STOP ===")