import tensorflow as tf
from data_loader import load_mnist
import numpy as np

def load_model_and_predict():
    model_path = "models/logistic_regression_model.keras"

    # Загружаем сохранённую модель
    model = tf.keras.models.load_model(model_path)
    print(f"Модель загружена из: {model_path}")

    # Загружаем данные
    _, _, X_test, y_test = load_mnist(flatten=True, normalize=True)

    # Предсказания для первых 10 изображений
    predictions = model.predict(X_test[:10])
    predicted_classes = np.argmax(predictions, axis=1)

    print("Предсказанные классы:", predicted_classes)
    print("Истинные классы     :", y_test[:10])

    # Расчёт точности на всем тестовом наборе
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Точность модели на тестовом наборе: {accuracy:.4f}")

if __name__ == "__main__":
    load_model_and_predict()