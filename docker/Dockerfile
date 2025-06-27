# Используем официальный образ TensorFlow с поддержкой GPU, Python 3.11
FROM tensorflow/tensorflow:2.19.0-gpu

# Рабочая директория внутри контейнера
WORKDIR /app

# Установим PYTHONPATH, чтобы работали абсолютные импорты из src
ENV PYTHONPATH=/app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# По умолчанию запускаем интерактивный bash
CMD ["/bin/bash"]