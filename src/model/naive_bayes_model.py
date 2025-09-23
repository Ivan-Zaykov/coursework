import os
from tqdm import tqdm
import joblib
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "Naive Bayes"
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.joblib")
BEST_PARAMS = {
    "alpha": 0.10292247147901223,
    "binarize": 0.5,
    "fit_prior": False
}
BATCH_SIZE = 1000  # единый размер батча для предсказания

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def train_naive_bayes():
    try:
        # Загружаем данные для обучения (без бинаризации здесь, она внутри модели)
        X_train, y_train, _, _ = load_mnist(flatten=True, binarize=False)

        if os.path.exists(MODEL_PATH):
            results_logger.info(f"Loading model from {MODEL_PATH} ...")
            clf = joblib.load(MODEL_PATH)
            results_logger.info("Model loaded, skipping training.")
            return clf

        # Создаем модель с оптимальными гиперпараметрами
        clf = BernoulliNB(
            alpha=BEST_PARAMS["alpha"],
            binarize=BEST_PARAMS["binarize"],
            fit_prior=BEST_PARAMS["fit_prior"]
        )

        results_logger.info("Training Naive Bayes model with optimized hyperparameters...")
        metric_logger.start("train")
        clf.fit(X_train, y_train)
        metric_logger.stop("train")

        elapsed_train = metric_logger.durations.get(f"{MODEL_NAME}_train", None)
        if elapsed_train is not None:
            results_logger.info(f"Training completed in {elapsed_train:.2f} seconds")

        joblib.dump(clf, MODEL_PATH)
        results_logger.info(f"Model saved to {MODEL_PATH}")
        return clf

    except Exception as e:
        error_logger.error(f"Error during training {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None

def evaluate_naive_bayes(clf):
    try:
        _, _, X_test, y_test = load_mnist(flatten=True, binarize=False)
        results_logger.info("Starting prediction...")
        metric_logger.start("evaluate")

        predictions = []
        for start_idx in tqdm(range(0, len(X_test), BATCH_SIZE), desc="Predicting"):
            end_idx = start_idx + BATCH_SIZE
            batch_preds = clf.predict(X_test[start_idx:end_idx])
            predictions.extend(batch_preds)
        predictions = predictions[:len(X_test)]

        metric_logger.stop("evaluate")

        accuracy = accuracy_score(y_test, predictions)
        metric_logger.set_accuracy(accuracy)

        results_logger.info(f"{MODEL_NAME} accuracy: {accuracy:.4f}")
        print(f"{MODEL_NAME} accuracy: {accuracy:.4f}")

        # Логируем confusion matrix
        metric_logger.log_confusion_matrix(y_test, predictions)

    except Exception as e:
        error_logger.error(f"Error during evaluation {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")

    metric_logger.start("total_run")
    model = train_naive_bayes()
    if model:
        evaluate_naive_bayes(model)
    metric_logger.stop("total_run")
    metric_logger.log_all()

    results_logger.info(f"=== {MODEL_NAME} STOP ===")
