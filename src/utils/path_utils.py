import os

# Абсолютный путь до корня проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Пути до важных директорий
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def ensure_dirs():
    """Создаёт необходимые директории, если их нет."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)