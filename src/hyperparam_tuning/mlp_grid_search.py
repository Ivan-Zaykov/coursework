import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

MODEL_NAME = "MLP"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def build_model(input_shape, num_classes, l2_reg=0.0):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
                  optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_evaluate(params, X_train, y_train, X_val, y_val):
    try:
        model = build_model(input_shape=(X_train.shape[1],),
                            num_classes=10,
                            l2_reg=params["l2_reg"]
                            )

        history = model.fit(
            X_train, y_train,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            verbose=0,
            validation_data=(X_val, y_val)
        )

        val_acc = history.history['val_accuracy'][-1]
        return params, val_acc
    except Exception as e:
        error_logger.error(f"Error with params {params}: {e}", exc_info=True)
        return params, None

def grid_search_mlp_parallel():
    X_train_full, y_train_full, X_test, y_test = load_mnist(flatten=True, normalize=True)

    # Уменьшаем размер обучающей выборки до 30000 для ускорения подбора
    X_train_full, y_train_full = shuffle(X_train_full, y_train_full, random_state=42)
    X_train_full = X_train_full[:30000]
    y_train_full = y_train_full[:30000]

    # Делим на 80% train и 20% validation
    split_idx = int(len(X_train_full) * 0.8)
    X_train, y_train = X_train_full[:split_idx], y_train_full[:split_idx]
    X_val, y_val = X_train_full[split_idx:], y_train_full[split_idx:]

    param_grid = {
        "batch_size": [32, 64, 128],
        "epochs": [5, 10],
        "l2_reg": [0.0, 0.01],
    }

    grid = list(ParameterGrid(param_grid))

    best_accuracy = 0.0
    best_params = None

    results_logger.info(f"Starting MLP hyperparameter grid search with {len(grid)} combinations on 30k samples (80% train / 20% val)")

    # Параллельное выполнение с количеством потоков 2-4 для баланса скорости и ресурсов
    max_workers = 3

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_evaluate, params, X_train, y_train, X_val, y_val) for params in grid]

        for future in as_completed(futures):
            params, val_acc = future.result()
            if val_acc is not None:
                metric_logger.set_hyperparameters(params, val_acc)
                result_msg = f"Params: {params}, Validation Accuracy: {val_acc:.4f}"
                print(result_msg)
                results_logger.info(result_msg)

                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_params = params

    metric_logger.log_all()

    results_logger.info(f"Best params: {best_params}, Validation Accuracy: {best_accuracy:.4f}")
    print(f"Best params: {best_params}, Validation Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} GridSearch START ===")
    start_time = time.time()
    grid_search_mlp_parallel()
    elapsed = time.time() - start_time
    results_logger.info(f"=== {MODEL_NAME} GridSearch STOP. Elapsed time: {elapsed:.2f} seconds ===")