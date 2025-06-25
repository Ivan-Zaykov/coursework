import os
import logging
import time
import tensorflow as tf
from data_loader import load_mnist

LOG_DIR = "logs"
RESULTS_LOG = os.path.join(LOG_DIR, "model_results.log")
ERRORS_LOG = os.path.join(LOG_DIR, "model_errors.log")
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_LOG),
        logging.StreamHandler()
    ]
)
results_logger = logging.getLogger('results_logger')

error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(ERRORS_LOG)
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

def train_logistic_regression_tf(epochs=10, batch_size=128):
    try:
        # Загружаем данные (flatten=True, normalize=True)
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

        # Создаём модель: логистическая регрессия = один слой Dense без скрытых слоёв
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),  # 28x28 → 784
            tf.keras.layers.Dense(10, activation='softmax')  # 10 выходов (классы)
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        start_time = time.time()

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        elapsed_time = time.time() - start_time
        result_msg = f"TF Logistic Regression accuracy: {accuracy:.4f}, Time elapsed: {elapsed_time:.2f} sec"
        print(result_msg)
        results_logger.info(f"Model: TF Logistic Regression, Accuracy: {accuracy:.4f}, Time: {elapsed_time:.2f} sec")

        # Сохраняем модель
        model_path = os.path.join(MODEL_DIR, "logistic_regression_model.keras")
        model.save(model_path)
        results_logger.info(f"Model saved to {model_path}")

    except Exception as e:
        error_logger.error(f"Error in TF Logistic Regression model: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    train_logistic_regression_tf()