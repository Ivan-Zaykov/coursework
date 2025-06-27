import logging
import os
from src.utils.path_utils import LOG_DIR, ensure_dirs

# Абсолютные пути к директориям logs и models (относительно src/utils/...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def setup_loggers():
    ensure_dirs()

    results_log_path = os.path.join(LOG_DIR, "model_results.log")
    errors_log_path = os.path.join(LOG_DIR, "model_errors.log")

    # Логгер результатов
    results_logger = logging.getLogger("results_logger")
    if not results_logger.hasHandlers():
        results_logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        file_handler = logging.FileHandler(results_log_path)
        file_handler.setFormatter(formatter)
        results_logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        results_logger.addHandler(stream_handler)

    # Логгер ошибок
    error_logger = logging.getLogger("error_logger")
    if not error_logger.hasHandlers():
        error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(errors_log_path)
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        error_logger.addHandler(error_handler)

    return results_logger, error_logger