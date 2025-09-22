import os
from itertools import product
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger
import numpy as np

MODEL_NAME = "SVM"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def grid_search_svm_all(max_samples=10000, cv_folds=3):
    try:
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

        # Ограничиваем выборку для ускорения подбора параметров
        X_train_small = X_train[:max_samples]
        y_train_small = y_train[:max_samples]

        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]  # Влияет только на rbf kernel
        }

        all_combinations = list(product(param_grid["C"], param_grid["kernel"], param_grid["gamma"]))
        total = len(all_combinations)
        results_logger.info(f"Total combinations to try: {total}")

        best_accuracy = 0
        best_params = None

        for idx, (C_val, kernel_val, gamma_val) in enumerate(all_combinations, 1):
            params = {"C": C_val, "kernel": kernel_val, "gamma": gamma_val}
            clf = svm.SVC(**params)

            # CV accuracy
            scores = cross_val_score(clf, X_train_small, y_train_small, cv=cv_folds, n_jobs=-1)
            mean_acc = np.mean(scores)

            # Логируем в MetricLogger каждую комбинацию
            metric_logger.set_hyperparameters(params, mean_acc)

            msg = f"[{idx}/{total}] Params: {params}, CV Accuracy: {mean_acc:.4f}"
            print(msg)
            results_logger.info(msg)

            if mean_acc > best_accuracy:
                best_accuracy = mean_acc
                best_params = params

        # Финальная запись всех логов и CSV
        metric_logger.log_all()

        results_logger.info(f"Best params: {best_params}, Best CV accuracy: {best_accuracy:.4f}")
        print(f"Best params: {best_params}, Best CV accuracy: {best_accuracy:.4f}")

        return best_params, best_accuracy

    except Exception as e:
        error_logger.error(f"Error during manual GridSearch for {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при подборе гиперпараметров: {e}")
        return None, None


if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} Manual GridSearch START ===")
    grid_search_svm_all()
    results_logger.info(f"=== {MODEL_NAME} Manual GridSearch STOP ===")