from src.utils.metric_logger import MetricLogger
import os
import tensorflow as tf
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR

MODEL_NAME = "TF Logistic Regression"

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def train_logistic_regression_tf(epochs=10, batch_size=128):
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, normalize=True)
        MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.keras")

        if os.path.exists(MODEL_PATH):
            results_logger.info(f"Loading model from {MODEL_PATH} ...")
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            metric_logger.start("training")
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
            metric_logger.stop("training")

            model.save(MODEL_PATH)
            results_logger.info(f"Model saved to {MODEL_PATH}")
            print("Обучение завершено")

        return model

    except Exception as e:
        error_logger.error(f"Error in training TF Logistic Regression model: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None


def evaluate_logistic_regression_tf(model):
    try:
        _, _, X_test, y_test = load_mnist(flatten=True, normalize=True)

        metric_logger.start("evaluation")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        metric_logger.set_accuracy(accuracy)

        metric_logger.stop("evaluation")

        result_msg = f"TF Logistic Regression accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

    except Exception as e:
        error_logger.error(f"Error in evaluating TF Logistic Regression model: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")


if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")

    metric_logger.start("total_run")
    model = train_logistic_regression_tf()
    if model:
        evaluate_logistic_regression_tf(model)
    metric_logger.stop("total_run")
    metric_logger.log_all()

    results_logger.info(f"=== {MODEL_NAME} STOP ===")