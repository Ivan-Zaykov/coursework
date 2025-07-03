# Исследование основных методов распознавания рукописных цифр на наборе данных MNIST с использованием библиотеки TensorFlow

Проект посвящён обучению и сравнению различных моделей машинного обучения на датасете **MNIST**

## Основные возможности

- **Классификация MNIST** с помощью:
  - Метод k ближайших соседей (k-NN)
  - Наивный байесовский метод
  - Логистическая регрессия
  - Случайного леса (Random Forest)
  - Метод опорных векторов (SVM)
  - Многослойный перцептрон (MLP)
  - Регуляризованная MLP (Dropout, BatchNorm, L2)
  - Сверточная нейронная сеть (CNN)

- **Поиск гиперпараметров** с помощью grid search и random search
- **Логгирование**:
  - Accuracy, confusion matrix
  - Время обучения и предсказания
  - Использование ресурсов (CPU, RAM, GPU)
  - CSV-отчёты и графики

## Структура проекта
```
.
├── docker                        # Файлы для Docker-окружения
│   └── Dockerfile               # Инструкция сборки образа
├── docker-compose.yml           # Композиция контейнеров (например, app + jupyter)
├── main.py                      # Точка входа в проект
├── README.md                    # Описание проекта
├── requirements.txt             # Список зависимостей (pip)
└── src                          # Основной исходный код проекта
    ├── hyperparam_tuning        # Скрипты для подбора гиперпараметров разных моделей
    │   ├── bayes_randomized_search_cv.py
    │   ├── knn_grid_search.py
    │   ├── logistic_regression_grid_search.py
    │   ├── mlp_grid_search.py
    │   ├── mlp_regularized_grid_search.py
    │   ├── random_foreset_grid_search.py
    │   └── svm_grid_search.py
    ├── model                    # Реализация моделей машинного обучения
    │   ├── cnn_model.py
    │   ├── k_nearest_neighbors.py
    │   ├── logistic_regression_model.py
    │   ├── mlp_model.py
    │   ├── mlp_regularized_model.py
    │   ├── naive_bayes_model.py
    │   ├── random_forest_model.py
    │   └── svm_model.py
    ├── preprocessing            # Предобработка входных данных
    │   ├── binarizer.py
    ├── utils                    # Вспомогательные утилиты: логгеры, загрузка данных и пр.
    │   ├── data_loader.py
    │   ├── logger_setup.py
    │   ├── metric_logger.py
    │   └── path_utils.py
    └── __init__.py              # Объявление пакета src
```


## Используемые технологии

- **Язык**: Python 3.11
- **Фреймворки**:
  - [TensorFlow 2.19](https://www.tensorflow.org/)
  - [Scikit-learn 1.7](https://scikit-learn.org/)
  - Matplotlib, Seaborn
  - tqdm, psutil, pynvml

## Запуск с помощью Docker

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/Ivan-Zaykov/coursework.git
cd coursework
```

### 2. Постройте Docker-образ
```
docker compose build
```

### 3. Запустите контейнер
```
docker compose up -d
```
### 4.  Вход в контейнер (интерактивный режим)
```
docker exec -it coursework-tf-gpu-1 bash
```
### 5. Запуск скриптов
```
# Пример: подбор гиперпараметров для k-NN
python src/hyperparam_tuning/knn_grid_search.py

# Пример: подбор гиперпараметров для логистической регрессии
python src/hyperparam_tuning/logistic_regression_grid_search.py

# Пример: запуск модели MLP
python src/model/mlp_model.py
```