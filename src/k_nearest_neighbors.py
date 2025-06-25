import logging
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from data_loader import load_mnist

# Путь к папке logs в корне проекта (предполагается, что скрипт запускается из корня)
LOG_DIR = "logs"
LOG_FILE = "model_results.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Создаём папку logs, если её нет
os.makedirs(LOG_DIR, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def train_and_evaluate_knn(k=3):
    X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    result_msg = f"k-NN accuracy (k={k}): {accuracy:.4f}"
    print(result_msg)

    logging.info(f"Model: k-NN, k={k}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_and_evaluate_knn()