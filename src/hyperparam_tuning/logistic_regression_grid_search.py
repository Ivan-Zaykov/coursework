import tensorflow as tf
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "Logistic Regression GridSearch"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def build_logistic_model(input_shape, learning_rate=0.001, l2_reg=0.0):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(
            10,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
        )
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def grid_search_logistic_regression():
    try:
        X_train, y_train, X_test, y_test = load_mnist(flatten=True, normalize=True)

        # Сетка параметров (включает значения по умолчанию)
        param_grid = {
            "learning_rate": [0.001, 0.0005],
            "l2_reg": [0.0, 0.001],
            "batch_size": [128, 64],
            "epochs": [10, 15]
        }

        grid = list(ParameterGrid(param_grid))
        results_logger.info(f"Total combinations: {len(grid)}")

        best_accuracy = 0
        best_params = None

        for params in grid:
            model = build_logistic_model(
                input_shape=(784,),
                learning_rate=params["learning_rate"],
                l2_reg=params["l2_reg"]
            )

            model.fit(
                X_train, y_train,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                verbose=0
            )

            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            acc = accuracy_score(y_test, y_pred)
            metric_logger.set_hyperparameters(params, acc)

            result_msg = f"Params: {params}, Accuracy: {acc:.4f}"
            print(result_msg)
            results_logger.info(result_msg)

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = params

        metric_logger.log_all()

        print(f"Best params: {best_params}, Accuracy: {best_accuracy:.4f}")
        results_logger.info(f"Best params: {best_params}, Accuracy: {best_accuracy:.4f}")

    except Exception as e:
        error_logger.error(f"Error in grid search: {e}", exc_info=True)
        print(f"Ошибка в подборе гиперпараметров: {e}")

if __name__ == "__main__":
    grid_search_logistic_regression()
