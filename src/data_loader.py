import tensorflow as tf
import numpy as np

def load_mnist(normalize=True, flatten=True):
    """
    Загружает датасет MNIST и подготавливает его.

    :param normalize: Нормализовать значения пикселей в диапазон [0,1]
    :param flatten: Преобразовать 28x28 изображения в вектор размером 784
    :return: Кортеж из обучающих и тестовых данных (X_train, y_train, X_test, y_test)
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

    return X_train, y_train, X_test, y_test
