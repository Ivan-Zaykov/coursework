from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from src.utils.data_loader import load_mnist
from src.utils.logger_setup import setup_loggers
from src.utils.metric_logger import MetricLogger

MODEL_NAME = "Naive Bayes"
results_logger, error_logger = setup_loggers()
metric_logger = MetricLogger(results_logger, model_name=MODEL_NAME)

def randomized_search_naive_bayes():
    try:
        X_train, y_train, _, _ = load_mnist(flatten=True, binarize=True)

        # Определяем распределения гиперпараметров
        param_dist = {
            "alpha": stats.uniform(0, 5),          # alpha от 0 до 5 (неотрицательное сглаживание)
            "binarize": [0.0, 0.5, 1.0, None],     # порог бинаризации
            "fit_prior": [True, False],            # использовать априорные вероятности
        }

        clf = BernoulliNB()

        random_search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_dist,
            n_iter=20,                # количество случайных комбинаций
            cv=3,                     # кросс-валидация 3-фолд
            scoring="accuracy",
            n_jobs=-1,                # параллельный запуск на всех ядрах
            verbose=2,
            random_state=42
        )

        results_logger.info(f"Starting RandomizedSearchCV for {MODEL_NAME}...")

        random_search.fit(X_train, y_train)

        results_logger.info(f"RandomizedSearchCV finished.")

        best_params = random_search.best_params_
        best_score = random_search.best_score_

        results_logger.info(f"Best params: {best_params}, Best accuracy: {best_score:.4f}")
        print(f"Best params: {best_params}, Best accuracy: {best_score:.4f}")

        # Записываем в metric_logger только лучшие параметры и accuracy
        metric_logger.set_hyperparameters(best_params, best_score)

        # Сохраняем логи и CSV файлы
        metric_logger.log_all()

    except Exception as e:
        error_logger.error(f"Error during RandomizedSearchCV for {MODEL_NAME}: {e}", exc_info=True)
        print(f"Произошла ошибка при подборе гиперпараметров: {e}")

if __name__ == "__main__":
    randomized_search_naive_bayes()