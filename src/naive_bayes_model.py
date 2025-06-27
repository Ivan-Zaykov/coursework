import os
import logging
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from data_loader import load_mnist
from tqdm import tqdm
import joblib

# Пути
LOG_DIR = "logs"
RESULTS_LOG = os.path.join(LOG_DIR, "model_results.log")
ERRORS_LOG = os.path.join(LOG_DIR, "model_errors.log")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.joblib")

# Создание директорий
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Настройка логирования
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

def train_and_evaluate_naive_bayes():
    try:
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, binarize=True)

        if os.path.exists(MODEL_PATH):
            print(f"Загрузка модели из {MODEL_PATH} ...")
            clf = joblib.load(MODEL_PATH)
            elapsed_train = "N/A (модель загружена)"
        else:
            clf = BernoulliNB()
            print("Обучение Naive Bayes...")
            start_time = time.time()
            clf.fit(X_train, y_train)
            elapsed_train = time.time() - start_time
            print(f"Обучение завершено за {elapsed_train:.2f} секунд")
            joblib.dump(clf, MODEL_PATH)
            print(f"Модель сохранена в {MODEL_PATH}")

        print("Предсказание...")
        predictions = []
        batch_size = 1000
        for start_idx in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
            end_idx = start_idx + batch_size
            batch_preds = clf.predict(X_test[start_idx:end_idx])
            predictions.extend(batch_preds)
        predictions = predictions[:len(X_test)]

        accuracy = accuracy_score(y_test, predictions)
        results_logger.info(f"Naive Bayes accuracy: {accuracy:.4f}, Training time: {elapsed_train}")
        print(f"Naive Bayes accuracy: {accuracy:.4f}")

    except Exception as e:
        error_logger.error(f"Error in Naive Bayes model: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    train_and_evaluate_naive_bayes()
