import os
import tensorflow as tf
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "MLP Multi-Layer Perceptron"

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def build_mlp_model(input_shape=(784,), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_mlp(epochs=10, batch_size=128):
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, normalize=True)
        model_path = os.path.join(MODEL_DIR, "mlp_model.keras")

        if os.path.exists(model_path):
            results_logger.info(f"Loading model from {model_path} ...")
            model = tf.keras.models.load_model(model_path)
        else:
            model = build_mlp_model()
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            metric_logger.start("training")
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
            metric_logger.stop("training")

            model.save(model_path)
            results_logger.info(f"Model saved to {model_path}")
            print("Обучение завершено")

        return model

    except Exception as e:
        error_logger.error(f"Error in training MLP model: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None

def evaluate_mlp(model):
    try:
        _, _, X_test, y_test = load_mnist(flatten=True, normalize=True)

        metric_logger.start("evaluation")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        metric_logger.stop("evaluation")

        result_msg = f"MLP accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

        # Запишем accuracy в MetricLogger
        metric_logger.set_accuracy(accuracy)

    except Exception as e:
        error_logger.error(f"Error in evaluating MLP model: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")

    metric_logger.start("total_run")
    model = train_mlp()
    if model:
        evaluate_mlp(model)
    metric_logger.stop("total_run")
    metric_logger.log_all()

    results_logger.info(f"=== {MODEL_NAME} STOP ===")
