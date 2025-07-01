import datetime
import time
import psutil
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
except pynvml.NVMLError:
    GPU_AVAILABLE = False

class MetricLogger:
    def __init__(self, logger, model_name):
        self.start_times = {}
        self.durations = {}
        self.logger = logger
        self.model_name = model_name

        # Для ресурсов будем хранить списки замеров между start и stop
        self.cpu_usages = {}
        self.ram_usages = {}
        self.gpu_usages = {}
        self.accuracy_scores = {}

        # Для гиперпараметров: список словарей, каждый со значениями параметров и accuracy
        self.hyperparams_log = []

        # Папка для текущего запуска
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_log_dir = os.path.join("logs", self.model_name.replace(" ", "_"), self.timestamp)
        os.makedirs(self.base_log_dir, exist_ok=True)

    def set_accuracy(self, accuracy: float):
        """Установить accuracy для total_run"""
        key = f"{self.model_name}_total_run"
        self.accuracy_scores[key] = accuracy

    def set_hyperparameters(self, hyperparams: dict, accuracy: float):
        """
        Добавить запись с гиперпараметрами и accuracy.

        :param hyperparams: dict с гиперпараметрами, например {"lr": 0.01, "batch_size": 32}
        :param accuracy: float - accuracy для этих гиперпараметров
        """
        entry = hyperparams.copy()
        entry["accuracy"] = accuracy
        self.hyperparams_log.append(entry)

        # Логируем в текстовом виде
        params_str = ", ".join(f"{k}={v}" for k, v in hyperparams.items())
        self.logger.info(f"Hyperparams: {params_str}, Accuracy: {accuracy:.4f}")

    def start(self, metric_name: str):
        key = f"{self.model_name}_{metric_name}"
        self.start_times[key] = time.perf_counter()
        self.cpu_usages[key] = []
        self.ram_usages[key] = []
        self.gpu_usages[key] = []
        self.logger.info(f"Started metric '{key}'")

        # Сделаем замер сразу при старте
        self._sample_resources(key)

    def stop(self, metric_name: str):
        key = f"{self.model_name}_{metric_name}"
        if key not in self.start_times:
            self.logger.warning(f"Timer '{key}' was not started.")
            return

        elapsed = time.perf_counter() - self.start_times[key]
        self.durations[key] = elapsed

        # Последний замер ресурсов при стопе
        self._sample_resources(key)

        # Теперь определим пиковые значения за период (start-stop)
        max_cpu = max(self.cpu_usages[key]) if self.cpu_usages[key] else 0
        max_ram = max(self.ram_usages[key]) if self.ram_usages[key] else 0
        # Фильтруем None из gpu_usages
        gpu_values = [v for v in self.gpu_usages[key] if v is not None]
        max_gpu = max(gpu_values) if gpu_values else None

        self.logger.info(
            f"Stopped metric '{key}', duration: {elapsed:.4f} sec, "
            f"peak CPU: {max_cpu:.1f}%, peak RAM: {max_ram / (1024 ** 2):.1f} MB" +
            (f", peak GPU: {max_gpu:.1f}%" if max_gpu is not None else "")
        )

    def log_all(self):
        headers = ["Metric", "Duration (sec)", "Peak CPU (%)", "Peak RAM (MB)", "Peak GPU (%)", "Accuracy"]
        rows = []

        for key, duration in self.durations.items():
            max_cpu = max(self.cpu_usages.get(key, [0]))
            max_ram = max(self.ram_usages.get(key, [0]))
            gpu_values = [v for v in self.gpu_usages.get(key, []) if v is not None]
            max_gpu = max(gpu_values) if gpu_values else None
            accuracy = self.accuracy_scores.get(key, "—")

            row = [
                key,
                f"{duration:.4f}",
                f"{max_cpu:.1f}",
                f"{max_ram / (1024 ** 2):.1f}",
                f"{max_gpu:.1f}" if max_gpu is not None else "N/A",
                f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy
            ]
            rows.append(row)

        col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]

        def format_row(row):
            return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

        table_lines = [
            f"=== Метрики производительности и времени {self.model_name} ===",
            format_row(headers),
            "-+-".join("-" * w for w in col_widths),
        ]
        table_lines.extend(format_row(row) for row in rows)

        self.logger.info("\n" + "\n".join(table_lines))

        # Сохраняем CSV файл с метриками
        csv_path = os.path.join(self.base_log_dir, "metrics.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
        self.logger.info(f"Metrics CSV saved to {csv_path}")

        # Сохраняем CSV гиперпараметров, если есть
        if self.hyperparams_log:
            hp_csv_path = os.path.join(self.base_log_dir, "hyperparameters.csv")
            with open(hp_csv_path, mode="w", newline="", encoding="utf-8") as f:
                keys = set()
                for entry in self.hyperparams_log:
                    keys.update(entry.keys())
                keys = sorted(keys)
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for entry in self.hyperparams_log:
                    writer.writerow(entry)
            self.logger.info(f"Hyperparameters CSV saved to {hp_csv_path}")

    def _sample_resources(self, key):
        # CPU usage (процент от 0 до 100)
        cpu = psutil.cpu_percent(interval=None)
        self.cpu_usages[key].append(cpu)

        # RAM usage в байтах (RSS используемой памяти процесса)
        process = psutil.Process()
        ram = process.memory_info().rss
        self.ram_usages[key].append(ram)

        # GPU usage через pynvml (если доступна)
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # первая видеокарта
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu = util.gpu  # процент загрузки GPU
                self.gpu_usages[key].append(gpu)
            except Exception:
                self.gpu_usages[key].append(None)
        else:
            # Если нет GPU или pynvml - None
            self.gpu_usages[key].append(None)

    def log_confusion_matrix(self, y_true, y_pred, labels=None, save_dir=None):
        """
        Логирует confusion matrix в текст и сохраняет картинку в файл с датой и именем модели.

        :param y_true: истинные метки
        :param y_pred: предсказанные метки
        :param labels: список меток классов (по умолчанию None)
        :param save_dir: директория для сохранения картинки (если None — не сохраняет)
        """
        labels = labels if labels is not None else list(range(10))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        num_classes = len(cm)

        # Текстовая таблица — без изменений
        header = [""] + [str(lbl) for lbl in labels]
        rows = []
        for i, row in enumerate(cm):
            rows.append([str(header[i + 1])] + [str(x) for x in row])

        col_widths = [max(len(str(x)) for x in col) for col in zip(*([header] + rows))]

        def format_row(row):
            return " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

        table_lines = [
            f"=== Confusion Matrix for {self.model_name} ===",
            format_row(header),
            "-+-".join("-" * w for w in col_widths),
        ]
        table_lines.extend(format_row(row) for row in rows)
        self.logger.info("\n" + "\n".join(table_lines))

        # Если save_dir не передан, сохраняем в базовую папку запуска
        save_dir = save_dir or self.base_log_dir
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_model_name = self.model_name.replace(" ", "_").replace("/", "_")
        filename = f"confusion_matrix_{safe_model_name}_{timestamp}.png"
        save_path = os.path.join(save_dir, filename)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Предсказанный класс")
        plt.ylabel("Истинный класс")
        plt.title(f"Confusion Matrix: {self.model_name}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        self.logger.info(f"Confusion matrix image saved to {save_path}")