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

def train_random_forest(n_estimators=100):
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, normalize=True)

        if os.path.exists(MODEL_PATH):
            results_logger.info(f"Loading model from {MODEL_PATH} ...")
            clf = joblib.load(MODEL_PATH)
        else:
            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            results_logger.info("Training Random Forest model...")
            metric_logger.start("training")
            clf.fit(X_train, y_train)
            metric_logger.stop("training")

            joblib.dump(clf, MODEL_PATH)
            results_logger.info(f"Model saved to {MODEL_PATH}")
            print("Обучение завершено")

        return clf

    except Exception as e:
        error_logger.error(f"Error in training Random Forest model: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None


def evaluate_random_forest(clf):
    try:
        _, _, X_test, y_test = load_mnist(flatten=True, normalize=True)

        results_logger.info("Начинается предсказание...")
        metric_logger.start("evaluation")

        predictions = []
        batch_size = 1000
        for start_idx in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
            end_idx = start_idx + batch_size
            batch_preds = clf.predict(X_test[start_idx:end_idx])
            predictions.extend(batch_preds)
        predictions = predictions[:len(X_test)]

        metric_logger.stop("evaluation")

        accuracy = accuracy_score(y_test, predictions)
        metric_logger.set_accuracy(accuracy)

        result_msg = f"Random Forest accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

        # Логируем confusion matrix
        metric_logger.log_confusion_matrix(y_test, predictions)
    except Exception as e:
        error_logger.error(f"Error in evaluating Random Forest model: {e}", exc_info=True)
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
