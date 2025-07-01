import os
import tensorflow as tf
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "CNN MNIST"

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def train_cnn(epochs=5, batch_size=128):
    try:
        # Загружаем данные, normalize=True, flatten=False — важен 2D формат для CNN
        X_train, y_train, X_test, y_test = load_mnist(flatten=False, normalize=True)

        # Добавляем размерность канала (MNIST grayscale = 1)
        X_train = X_train[..., tf.newaxis]
        X_test = X_test[..., tf.newaxis]

        model_path = os.path.join(MODEL_DIR, "cnn_model.keras")

        if os.path.exists(model_path):
            results_logger.info(f"Loading model from {model_path} ...")
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
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

            model.save(model_path)
            results_logger.info(f"Model saved to {model_path}")
            print("Обучение завершено")

        return model, X_test, y_test

    except Exception as e:
        error_logger.error(f"Error in training CNN model: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None, None, None


def evaluate_cnn(model, X_test, y_test):
    try:
        metric_logger.start("evaluation")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        metric_logger.stop("evaluation")

        result_msg = f"CNN MNIST accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

        # Запишем accuracy в MetricLogger
        metric_logger.set_accuracy(accuracy)

    except Exception as e:
        error_logger.error(f"Error in evaluating CNN model: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")


if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")

    metric_logger.start("total_run")
    model, X_test, y_test = train_cnn()
    if model is not None:
        evaluate_cnn(model, X_test, y_test)
    metric_logger.stop("total_run")
    metric_logger.log_all()

    results_logger.info(f"=== {MODEL_NAME} STOP ===")