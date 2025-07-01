import os
import tensorflow as tf
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.path_utils import MODEL_DIR
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "MLP Regularized"

results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)


def train_mlp_regularized(epochs=10, batch_size=128):
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, normalize=True)
        MODEL_PATH = os.path.join(MODEL_DIR, "mlp_regularized_model.keras")

        if os.path.exists(MODEL_PATH):
            results_logger.info(f"Loading model from {MODEL_PATH} ...")
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            metric_logger.start("training")
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
            metric_logger.stop("training")

            model.save(MODEL_PATH)
            results_logger.info(f"Model saved to {MODEL_PATH}")
            print("Обучение завершено")

        return model

    except Exception as e:
        error_logger.error(f"Error in training MLP Regularized model: {e}", exc_info=True)
        print(f"Произошла ошибка при обучении: {e}")
        return None


def evaluate_mlp_regularized(model):
    try:
        _, _, X_test, y_test = load_mnist(flatten=True, normalize=True)

        metric_logger.start("evaluation")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        metric_logger.stop("evaluation")
        metric_logger.set_accuracy(accuracy)

        result_msg = f"{MODEL_NAME} accuracy: {accuracy:.4f}"
        print(result_msg)
        results_logger.info(result_msg)

        # Получаем предсказания вероятностей
        y_pred_probs = model.predict(X_test)
        # Конвертируем вероятности в метки классов
        y_pred_labels = y_pred_probs.argmax(axis=1)

        # Логируем confusion matrix
        metric_logger.log_confusion_matrix(y_test, y_pred_labels)
    except Exception as e:
        error_logger.error(f"Error in evaluating MLP Regularized model: {e}", exc_info=True)
        print(f"Произошла ошибка при оценке: {e}")


if __name__ == "__main__":
    results_logger.info(f"=== {MODEL_NAME} START ===")
    metric_logger.start("total_run")

    model = train_mlp_regularized()
    if model:
        evaluate_mlp_regularized(model)

    metric_logger.stop("total_run")
    metric_logger.log_all()
    results_logger.info(f"=== {MODEL_NAME} STOP ===")