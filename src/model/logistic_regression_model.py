from src.utils.metric_logger import MetricLogger
import os
import tensorflow as tf
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR

MODEL_NAME = "TF Logistic Regression"

# Подобранные гиперпараметры
BEST_PARAMS = {
    "batch_size": 128,
    "epochs": 15,
    "learning_rate": 0.001,
    "l2_reg": 0.0
}

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def build_logistic_regression_model(input_shape=784, l2_reg=0.0, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(10, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_logistic_regression_tf():
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, normalize=True)
        MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.keras")

        if os.path.exists(MODEL_PATH):
            results_logger.info(f"Loading model from {MODEL_PATH} ...")
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            model = build_logistic_regression_model(l2_reg=BEST_PARAMS["l2_reg"],
                                                    learning_rate=BEST_PARAMS["learning_rate"])

            metric_logger.start("train")
            model.fit(X_train, y_train,
                      epochs=BEST_PARAMS["epochs"],
                      batch_size=BEST_PARAMS["batch_size"],
                      verbose=2)
            metric_logger.stop("train")

            model.save(MODEL_PATH)
            results_logger.info(f"Model saved to {MODEL_PATH}")
            print("Обучение завершено")

        return model

    except Exception as e:
        error_logger.error(f"Error in training {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None

def evaluate_logistic_regression_tf(model):
    try:
        _, _, X_test, y_test = load_mnist(flatten=True, normalize=True)

        metric_logger.start("evaluate")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        metric_logger.set_accuracy(accuracy)

        y_pred_probs = model.predict(X_test)
        y_pred = y_pred_probs.argmax(axis=1)

        metric_logger.log_confusion_matrix(y_test, y_pred)
        metric_logger.stop("evaluate")

        result_msg = f"{MODEL_NAME} accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

    except Exception as e:
        error_logger.error(f"Error in evaluating {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")

    metric_logger.start("total_run")
    model = train_logistic_regression_tf()
    if model:
        evaluate_logistic_regression_tf(model)
    metric_logger.stop("total_run")
    metric_logger.log_all()

    results_logger.info(f"=== {MODEL_NAME} STOP ===")
