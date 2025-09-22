from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from itertools import product
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "Random Forest"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def grid_search_random_forest():
    try:
        # Загружаем данные
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

        # Используем подвыборку для ускорения подбора
        N_SUBSAMPLE = 10000
        X_subsample = X_train[:N_SUBSAMPLE]
        y_subsample = y_train[:N_SUBSAMPLE]

        # Параметры
        n_estimators_list = [50, 100, 150]
        max_depth_list = [10, 20, None]
        max_features_list = ["sqrt", "log2"]

        best_accuracy = 0
        best_params = None

        # Проходим по всем комбинациям
        total_combinations = len(n_estimators_list) * len(max_depth_list) * len(max_features_list)
        results_logger.info(f"Starting manual GridSearch over {total_combinations} combinations...")

        for n_estimators, max_depth, max_features in product(n_estimators_list, max_depth_list, max_features_list):
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": max_features,
                "random_state": 42,
                "n_jobs": -1
            }

            clf = RandomForestClassifier(**params)

            # Кросс-валидация на 3 фолдах
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in skf.split(X_subsample, y_subsample):
                X_tr, X_val = X_subsample[train_idx], X_subsample[val_idx]
                y_tr, y_val = y_subsample[train_idx], y_subsample[val_idx]

                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                cv_scores.append(acc)

            mean_acc = sum(cv_scores) / len(cv_scores)

            # Логируем эту комбинацию
            metric_logger.set_hyperparameters(params, mean_acc)
            results_logger.info(f"Params: {params}, CV Accuracy: {mean_acc:.4f}")
            print(f"Params: {params}, CV Accuracy: {mean_acc:.4f}")

            if mean_acc > best_accuracy:
                best_accuracy = mean_acc
                best_params = params

        # Тестовая точность на лучшей модели
        best_model = RandomForestClassifier(**best_params)
        best_model.fit(X_subsample, y_subsample)
        y_test_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        results_logger.info(f"Best params: {best_params}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Best params: {best_params}, Test Accuracy: {test_accuracy:.4f}")

        # Логируем тестовую точность и confusion matrix
        metric_logger.set_accuracy(test_accuracy)
        metric_logger.log_confusion_matrix(y_test, y_test_pred)
        metric_logger.log_all()

    except Exception as e:
        error_logger.error(f"Error during GridSearch for {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при подборе гиперпараметров: {e}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} GridSearch START ===")
    grid_search_random_forest()
    results_logger.info(f"=== {MODEL_NAME} GridSearch STOP ===")