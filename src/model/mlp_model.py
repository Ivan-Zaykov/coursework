import os
import tensorflow as tf
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "MLP"

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

# Подобранные гиперпараметры
BEST_PARAMS = {
    "batch_size": 64,
    "epochs": 10,
    "l2_reg": 0.0
}

def build_mlp_model(input_shape=(784,), num_classes=10, l2_reg=0.0):
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_mlp():
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, normalize=True)
        model_path = os.path.join(MODEL_DIR, "mlp_model.keras")

        if os.path.exists(model_path):
            results_logger.info(f"Loading model from {model_path} ...")
            model = tf.keras.models.load_model(model_path)
        else:
            model = build_mlp_model(l2_reg=BEST_PARAMS["l2_reg"])
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            results_logger.info("Training MLP model with optimized hyperparameters...")
            metric_logger.start("training")
            model.fit(X_train, y_train,
                      epochs=BEST_PARAMS["epochs"],
                      batch_size=BEST_PARAMS["batch_size"],
                      verbose=2)
            metric_logger.stop("training")

            model.save(model_path)
            results_logger.info(f"Model saved to {model_path}")
            print("Обучение завершено")

        return model

    except Exception as e:
        error_logger.error(f"Error in training MLP model: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None

def evaluate_mlp(model):
    try:
        _, _, X_test, y_test = load_mnist(flatten=True, normalize=True)

        metric_logger.start("evaluation")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        metric_logger.stop("evaluation")
        metric_logger.set_accuracy(accuracy)

        result_msg = f"MLP accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

        # Предсказания классов
        y_pred_labels = model.predict(X_test).argmax(axis=1)
        metric_logger.log_confusion_matrix(y_test, y_pred_labels)

    except Exception as e:
        error_logger.error(f"Error in evaluating MLP model: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")

    metric_logger.start("total_run")
    model = train_mlp()
    if model:
        evaluate_mlp(model)
    metric_logger.stop("total_run")
    metric_logger.log_all()

    results_logger.info(f"=== {MODEL_NAME} STOP ===")
