import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger
from src.utils.path_utils import MODEL_DIR

MODEL_NAME = "Random Forest"

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")

# Подобранные гиперпараметры
BEST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 20,
    "max_features": "log2",
    "random_state": 42,
    "n_jobs": -1
}

def train_random_forest():
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, normalize=True)

        if os.path.exists(MODEL_PATH):
            results_logger.info(f"Loading model from {MODEL_PATH} ...")
            clf = joblib.load(MODEL_PATH)
        else:
            clf = RandomForestClassifier(**BEST_PARAMS)

            results_logger.info("Training Random Forest model with optimized hyperparameters...")
            metric_logger.start("train")
            clf.fit(X_train, y_train)
            metric_logger.stop("train")

            joblib.dump(clf, MODEL_PATH)
            results_logger.info(f"Model saved to {MODEL_PATH}")
            print("Обучение завершено")

        return clf

    except Exception as e:
        error_logger.error(f"Error in training {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None


def evaluate_random_forest(clf):
    try:
        _, _, X_test, y_test = load_mnist(flatten=True, normalize=True)

        results_logger.info("Starting prediction...")
        metric_logger.start("evaluate")

        predictions = []
        batch_size = 1000
        for start_idx in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
            end_idx = start_idx + batch_size
            batch_preds = clf.predict(X_test[start_idx:end_idx])
            predictions.extend(batch_preds)
        predictions = predictions[:len(X_test)]

        metric_logger.stop("evaluate")

        accuracy = accuracy_score(y_test, predictions)
        metric_logger.set_accuracy(accuracy)
        metric_logger.log_confusion_matrix(y_test, predictions)

        result_msg = f"{MODEL_NAME} accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

    except Exception as e:
        error_logger.error(f"Error in evaluating {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")


if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")

    metric_logger.start("total_run")
    model = train_random_forest()
    if model:
        evaluate_random_forest(model)
    metric_logger.stop("total_run")
    metric_logger.log_all()

    results_logger.info(f"=== {MODEL_NAME} STOP ===")