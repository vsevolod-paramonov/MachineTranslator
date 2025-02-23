import logging
from datetime import datetime
import os


class Logger:

    def __init__(self, log_root: str=None, log_name: str=None):
        
        self.logger = logging.getLogger('Logger')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.handlers = {'console_handler': logging.StreamHandler()}

        if log_root:

            os.makedirs(log_root, exist_ok=True)  
            filename = datetime.now().strftime('%Y-%m-%d') if not log_name else log_name
            self.handlers['file_handler'] = logging.FileHandler(os.path.join(log_root, f'{filename}.log'), mode='w')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in self.handlers.values():
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
    def _log_metrics(self, metrics: dict, epoch: int):

        metrics_string = f'[Epoch {epoch}] ' + ', '.join(f'{key}: {value}' for key, value in metrics.items())
        self.logger.info(metrics_string)

    def _log_custom_message(self, message: str):
        self.logger.info(message)


