from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger
from tqdm import tqdm

MODEL_NAME = "k-NN GridSearch"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def grid_search_knn():
    X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

    param_grid = {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # 1 — Манхэттен, 2 — Евклид
    }

    best_accuracy = 0
    best_params = None

    total_combinations = (
        len(param_grid["n_neighbors"]) *
        len(param_grid["weights"]) *
        len(param_grid["p"])
    )

    results_logger.info(f"Starting GridSearch over {total_combinations} combinations")

    for k in param_grid["n_neighbors"]:
        for weight in param_grid["weights"]:
            for p_val in param_grid["p"]:
                params = {
                    "n_neighbors": k,
                    "weights": weight,
                    "p": p_val
                }

                knn = KNeighborsClassifier(**params)
                knn.fit(X_train, y_train)

                y_pred = knn.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                metric_logger.set_accuracy(accuracy)
                metric_logger.set_hyperparameters(params, accuracy)

                result_msg = f"Params: {params}, Accuracy: {accuracy:.4f}"
                print(result_msg)
                results_logger.info(result_msg)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params

    metric_logger.log_all()

    results_logger.info(f"Best params: {best_params}, Best accuracy: {best_accuracy:.4f}")
    print(f"Best params: {best_params}, Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} GridSearch START ===")
    grid_search_knn()
    results_logger.info(f"=== {MODEL_NAME} GridSearch STOP ===")