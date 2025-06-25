import logging
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from data_loader import load_mnist

LOG_DIR = "logs"
RESULTS_LOG = os.path.join(LOG_DIR, "model_results.log")
ERRORS_LOG = os.path.join(LOG_DIR, "model_errors.log")

os.makedirs(LOG_DIR, exist_ok=True)

# Логгер для результатов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_LOG),
        logging.StreamHandler()
    ]
)
results_logger = logging.getLogger('results_logger')

# Логгер для ошибок
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(ERRORS_LOG)
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

def train_and_evaluate_knn(k=3):
    try:
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        result_msg = f"k-NN accuracy (k={k}): {accuracy:.4f}"
        print(result_msg)
        results_logger.info(f"Model: k-NN, k={k}, Accuracy: {accuracy:.4f}")
    except Exception as e:
        error_logger.error(f"Error in k-NN model: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    train_and_evaluate_knn()