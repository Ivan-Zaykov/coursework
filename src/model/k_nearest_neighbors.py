from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers

results_logger, error_logger = setup_loggers()

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