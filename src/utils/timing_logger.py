import time

class TimingLogger:
    def __init__(self, logger, model_name):
        self.start_times = {}
        self.durations = {}
        self.logger = logger
        self.model_name = model_name

    def start(self, metric_name: str):
        key = f"{self.model_name}_{metric_name}"
        self.start_times[key] = time.perf_counter()
        self.logger.info(f"Started timer '{key}'")

    def stop(self, metric_name: str):
        key = f"{self.model_name}_{metric_name}"
        if key not in self.start_times:
            self.logger.warning(f"Timer '{key}' was not started.")
            return
        elapsed = time.perf_counter() - self.start_times[key]
        self.durations[key] = elapsed
        self.logger.info(f"Stopped timer '{key}', duration: {elapsed:.4f} seconds")

    def log_all(self):
        self.logger.info("All recorded timings:")
        for key, duration in self.durations.items():
            self.logger.info(f"{key}: {duration:.4f} seconds")