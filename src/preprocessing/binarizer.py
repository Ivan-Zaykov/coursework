def binarize_images(images, threshold=0.5, normalized=True):
    """
    Бинаризует изображения по порогу.

    :param images: numpy-массив изображений (возможно нормализованный)
    :param threshold: значение порога (по умолчанию 0.5)
    :param normalized: если False, сначала делим на 255
    :return: бинаризованные изображения
    """
    if not normalized:
        images = images / 255.0
    return (images > threshold).astype(float)
