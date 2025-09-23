from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "k-NN"

# Оптимальные гиперпараметры, подобранные через GridSearch
BEST_PARAMS = {
    "n_neighbors": 3,
    "weights": "distance",
    "p": 2
}

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def train_and_evaluate_knn():
    try:
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

        # Обучение модели
        metric_logger.start("train")
        knn = KNeighborsClassifier(**BEST_PARAMS)
        knn.fit(X_train, y_train)
        metric_logger.stop("train")

        # Предсказание и оценка
        metric_logger.start("evaluate")
        y_pred = knn.predict(X_test)
        metric_logger.stop("evaluate")

        accuracy = accuracy_score(y_test, y_pred)
        metric_logger.set_accuracy(accuracy)
        metric_logger.log_confusion_matrix(y_test, y_pred)

        results_logger.info(
            f"{MODEL_NAME} accuracy: {accuracy:.4f} | "
            f"Parameters: {BEST_PARAMS}"
        )
        print(f"{MODEL_NAME} accuracy: {accuracy:.4f}")

    except Exception as e:
        error_logger.error(f"Error in {MODEL_NAME} model: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")

    metric_logger.start("total_run")
    train_and_evaluate_knn()
    metric_logger.stop("total_run")
    metric_logger.log_all()

    results_logger.info(f"=== {MODEL_NAME} STOP ===")
