import os
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger
from src.utils.path_utils import MODEL_DIR

MODEL_NAME = "CNN Hyperparameter GridSearch"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def create_cnn_model(filters1, filters2, dense_units, l2_reg=0.0, dropout_rate=0.0):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters1, kernel_size=(3,3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters2, kernel_size=(3,3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_evaluate_cnn(params, X_train_part, y_train_part, X_val, y_val):
    try:
        model = create_cnn_model(
            filters1=params['filters1'],
            filters2=params['filters2'],
            dense_units=params['dense_units'],
            l2_reg=params['l2_reg'],
            dropout_rate=params['dropout_rate']
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
        error_logger.error(f"Error with params {params}: {e}", exc_info=True)
        return params, None

def grid_search_cnn_parallel():
    X_train_full, y_train_full, _, _ = load_mnist(flatten=False, normalize=True)
    X_train_full = X_train_full[..., tf.newaxis]  # добавляем канал
    X_train_full, y_train_full = shuffle(X_train_full, y_train_full, random_state=42)

    # Берем подвыборку для ускорения подбора
    X_train_full = X_train_full[:10000]
    y_train_full = y_train_full[:10000]

    # Разделяем train/validation 80/20
    split_idx = int(len(X_train_full) * 0.8)
    X_train_part, y_train_part = X_train_full[:split_idx], y_train_full[:split_idx]
    X_val, y_val = X_train_full[split_idx:], y_train_full[split_idx:]

    # Сетка гиперпараметров
    param_grid = {
        "filters1": [16, 32],
        "filters2": [32, 64],
        "dense_units": [64, 128],
        "l2_reg": [0.0, 0.001],
        "dropout_rate": [0.0, 0.2],
        "batch_size": [64, 128],
        "epochs": [5]  # увеличь при необходимости
    }

    grid = list(ParameterGrid(param_grid))
    total_combinations = len(grid)
    results_logger.info(f"Starting parallel CNN Grid Search with {total_combinations} combinations")

    best_accuracy = 0.0
    best_params = None
    max_workers = 2  # регулируем по ресурсам

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_evaluate_cnn, params, X_train_part, y_train_part, X_val, y_val): params
                   for params in grid}

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

    results_logger.info(f"Best params: {best_params}, Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Best params: {best_params}, Best Validation Accuracy: {best_accuracy:.4f}")

    # Сохраняем все логи
    metric_logger.log_all()

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} Grid Search START ===")
    grid_search_cnn_parallel()
    results_logger.info(f"=== {MODEL_NAME} Grid Search STOP ===")
