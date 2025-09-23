import os
import tensorflow as tf
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "CNN"

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

BEST_PARAMS = {
    "batch_size": 64,
    "epochs": 5,
    "filters1": 32,
    "filters2": 32,
    "dense_units": 128,
    "dropout_rate": 0.0,
    "l2_reg": 0.0
}

def build_cnn_model(input_shape=(28,28,1), num_classes=10):
    regularizer = tf.keras.regularizers.l2(BEST_PARAMS["l2_reg"]) if BEST_PARAMS["l2_reg"] > 0 else None
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(BEST_PARAMS["filters1"], kernel_size=(3,3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizer),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(BEST_PARAMS["filters2"], kernel_size=(3,3), activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(BEST_PARAMS["dense_units"], activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dropout(BEST_PARAMS["dropout_rate"]),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_cnn():
    try:
        X_train, y_train, X_test, y_test = load_mnist(flatten=False, normalize=True)
        X_train = X_train[..., tf.newaxis]
        X_test = X_test[..., tf.newaxis]

        model_path = os.path.join(MODEL_DIR, "cnn_model.keras")

        if os.path.exists(model_path):
            results_logger.info(f"Loading model from {model_path} ...")
            model = tf.keras.models.load_model(model_path)
        else:
            model = build_cnn_model()
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            results_logger.info("Training CNN model with optimized hyperparameters...")
            metric_logger.start("training")
            model.fit(X_train, y_train,
                      epochs=BEST_PARAMS["epochs"],
                      batch_size=BEST_PARAMS["batch_size"],
                      verbose=2)
            metric_logger.stop("training")

            model.save(model_path)
            results_logger.info(f"Model saved to {model_path}")
            print("Обучение завершено")

        return model, X_test, y_test

    except Exception as e:
        error_logger.error(f"Error in training CNN model: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None, None, None

def evaluate_cnn(model, X_test, y_test):
    try:
        metric_logger.start("evaluation")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        metric_logger.stop("evaluation")
        metric_logger.set_accuracy(accuracy)

        result_msg = f"{MODEL_NAME} accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

        y_pred_labels = model.predict(X_test).argmax(axis=1)
        metric_logger.log_confusion_matrix(y_test, y_pred_labels)

    except Exception as e:
        error_logger.error(f"Error in evaluating CNN model: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")

if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")
    metric_logger.start("total_run")

    model, X_test, y_test = train_cnn()
    if model:
        evaluate_cnn(model, X_test, y_test)

    metric_logger.stop("total_run")
    metric_logger.log_all()
    results_logger.info(f"=== {MODEL_NAME} STOP ===")
