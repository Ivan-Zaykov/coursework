import os
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib

from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR

results_logger, error_logger = setup_loggers()

MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")

def train_and_evaluate_svm():
    try:
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

        if os.path.exists(MODEL_PATH):
            print(f"Загрузка модели из {MODEL_PATH} ...")
            clf = joblib.load(MODEL_PATH)
            elapsed_train = "N/A (модель загружена)"
        else:
            clf = svm.SVC(kernel='linear')
            print("Обучение SVM...")
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
        results_logger.info(f"SVM accuracy: {accuracy:.4f}, Training time: {elapsed_train} sec")
        print(f"SVM accuracy: {accuracy:.4f}")

    except Exception as e:
        error_logger.error(f"Error in SVM model: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    train_and_evaluate_svm()