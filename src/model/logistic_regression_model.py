import time
import os
import tensorflow as tf
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR

results_logger, error_logger = setup_loggers()

def train_logistic_regression_tf(epochs=10, batch_size=128):
    try:
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

        MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.keras")

        if os.path.exists(MODEL_PATH):
            print(f"Загрузка модели из {MODEL_PATH} ...")
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

            start_time = time.time()
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
            elapsed_time = time.time() - start_time

            model.save(MODEL_PATH)
            results_logger.info(f"Model saved to {MODEL_PATH}")
            results_logger.info(f"Model trained and saved in {elapsed_time:.2f} seconds")
            print(f"Обучение завершено за {elapsed_time:.2f} секунд")

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        result_msg = f"TF Logistic Regression accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(f"Model: TF Logistic Regression, Accuracy: {accuracy:.4f}")

    except Exception as e:
        error_logger.error(f"Error in TF Logistic Regression model: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    train_logistic_regression_tf()