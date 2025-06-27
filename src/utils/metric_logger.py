import time
import psutil

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
        headers = ["Metric", "Duration (sec)", "Peak CPU (%)", "Peak RAM (MB)", "Peak GPU (%)"]
        # Формируем строки таблицы
        rows = []
        for key, duration in self.durations.items():
            max_cpu = max(self.cpu_usages.get(key, [0]))
            max_ram = max(self.ram_usages.get(key, [0]))
            gpu_values = [v for v in self.gpu_usages.get(key, []) if v is not None]
            max_gpu = max(gpu_values) if gpu_values else None

            row = [
                key,
                f"{duration:.4f}",
                f"{max_cpu:.1f}",
                f"{max_ram / (1024 ** 2):.1f}",
                f"{max_gpu:.1f}" if max_gpu is not None else "N/A"
            ]
            rows.append(row)

        # Вычислим ширину каждого столбца по максимальной длине элемента (с учетом заголовков)
        col_widths = []
        for i in range(len(headers)):
            max_len = max(len(headers[i]), max(len(row[i]) for row in rows) if rows else 0)
            col_widths.append(max_len)

        # Функция для форматирования строки с выравниванием по ширинам
        def format_row(row):
            return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

        # Сборка всей таблицы
        table_lines = []
        table_lines.append("=== Все метрики производительности и времени ===")
        table_lines.append(format_row(headers))
        table_lines.append("-+-".join("-" * w for w in col_widths))
        for row in rows:
            table_lines.append(format_row(row))

        table_text = "\n".join(table_lines)
        self.logger.info("\n" + table_text)

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