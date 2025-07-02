from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
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

        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [10, 20, None],
            "max_features": ["sqrt", "log2"]
        }

        clf = RandomForestClassifier(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="accuracy",
            verbose=2,
            n_jobs=-1
        )

        results_logger.info(f"Starting GridSearchCV on subsample of size {N_SUBSAMPLE}...")
        grid_search.fit(X_subsample, y_subsample)

        best_params = grid_search.best_params_
        results_logger.info(f"Best parameters: {best_params}")

        # Оцениваем на тестовой выборке
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results_logger.info(f"Test accuracy: {accuracy:.4f}")
        print(f"Best params: {best_params}, Accuracy: {accuracy:.4f}")

        # Логируем
        metric_logger.set_hyperparameters(best_params, accuracy)
        metric_logger.set_accuracy(accuracy)
        metric_logger.log_confusion_matrix(y_test, y_pred)
        metric_logger.log_all()

    except Exception as e:
        error_logger.error(f"Error during GridSearchCV for {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при подборе гиперпараметров: {e}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} GridSearch START ===")
    grid_search_random_forest()
    results_logger.info(f"=== {MODEL_NAME} GridSearch STOP ===")