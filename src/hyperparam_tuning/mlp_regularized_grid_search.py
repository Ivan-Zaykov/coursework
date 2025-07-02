import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "Regularized MLP"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def create_model(l2_reg, dropout_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_and_evaluate(params, X_train_part, y_train_part, X_val, y_val):
    try:
        model = create_model(params['l2_reg'], params['dropout_rate'])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history = model.fit(
            X_train_part, y_train_part,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            verbose=0
        )
        val_acc = history.history['val_accuracy'][-1]
        return params, val_acc
    except Exception as e:
        error_logger.error(f"Error training with params {params}: {e}", exc_info=True)
        return params, None

def grid_search_regularized_mlp_parallel():
    # Загружаем данные
    X_train_full, y_train_full, _, _ = load_mnist(flatten=True, normalize=True)

    # Перемешиваем и берем подвыборку
    X_train_full, y_train_full = shuffle(X_train_full, y_train_full, random_state=42)
    X_train, y_train = X_train_full[:30000], y_train_full[:30000]

    # Разделяем 80/20 на train и validation
    val_split = 0.2
    split_idx = int(len(X_train) * (1 - val_split))
    X_train_part, y_train_part = X_train[:split_idx], y_train[:split_idx]
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]

    param_grid = {
        'batch_size': [32, 64, 128],
        'epochs': [10],
        'l2_reg': [0.0, 0.01],
        'dropout_rate': [0.2, 0.3]
    }

    grid = list(ParameterGrid(param_grid))
    total_combinations = len(grid)
    results_logger.info(f"Starting parallel Grid Search with {total_combinations} combinations...")

    best_accuracy = 0.0
    best_params = None

    max_workers = 3  # Можно менять, например 2-4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_and_evaluate, params, X_train_part, y_train_part, X_val, y_val): params
            for params in grid
        }

        for future in as_completed(futures):
            params, val_acc = future.result()
            if val_acc is not None:
                metric_logger.set_accuracy(val_acc)
                metric_logger.set_hyperparameters(params, val_acc)

                result_msg = f"Params: {params}, Validation Accuracy: {val_acc:.4f}"
                print(result_msg)
                results_logger.info(result_msg)

                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_params = params

    results_logger.info(f"Best params: {best_params}, Best accuracy: {best_accuracy:.4f}")
    print(f"Best params: {best_params}, Best accuracy: {best_accuracy:.4f}")

    metric_logger.log_all()

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} Parallel Grid Search START ===")
    grid_search_regularized_mlp_parallel()
    results_logger.info(f"=== {MODEL_NAME} Parallel Grid Search STOP ===")