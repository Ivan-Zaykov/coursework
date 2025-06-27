import tensorflow as tf
from src.preprocessing.binarizer import binarize_images

def load_mnist(flatten=True, normalize=True, binarize=False, threshold=0.5):
    """
    Загружает датасет MNIST и подготавливает его.

    :param flatten: Преобразовать 28x28 изображения в вектор размером 784
    :param normalize: Нормализовать значения пикселей в диапазон [0,1]
    :param binarize: Бинаризовать значения пикселей (по порогу)
    :param threshold: Порог бинаризации (используется если binarize=True)
    :return: Кортеж из обучающих и тестовых данных (X_train, y_train, X_test, y_test)
    """
    # Загружаем данные MNIST из TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if binarize:
        # Бинаризация изображений с заданным порогом
        X_train = binarize_images(X_train, threshold=threshold, normalized=normalize)
        X_test = binarize_images(X_test, threshold=threshold, normalized=normalize)
    elif normalize:
        # Нормализация пикселей в диапазон [0,1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    if flatten:
        # Преобразуем изображения 28x28 в плоские векторы длиной 784
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

    return X_train, y_train, X_test, y_test
