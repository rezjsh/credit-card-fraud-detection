import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from credit_card_fraud_detection.core.singleton import SingletonMeta




class Logger(metaclass=SingletonMeta):
    """
    A singleton class for managing application-wide logging.
    Configures and provides a thread-safe, pre-configured logger instance.
    """
    def __init__(
        self, 
        logger_name: str = "credit_card_fraud_detection", 
        log_dir: str = "logs", 
        log_file_name: str = "running_logs.log",
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ):
        """
        Initializes the Logger singleton.

        Args:
            logger_name (str): Unique name for the logger object.
            log_dir (str): Directory where logs will be saved.
            log_file_name (str): Output filename.
            level (int): Logging threshold level (e.g., logging.DEBUG, logging.INFO).
            max_bytes (int): Maximum size of a single log file before it rolls over.
            backup_count (int): Number of rotated historical log files to retain.
        """
        self._logger = logging.getLogger(logger_name)
        
        # Prevent re-initialization if handlers are already configured
        if not self._logger.handlers:
            self._logger.setLevel(level)
            
            # CRITICAL: Prevent logs from bubbling up to the root logger.
            # This stops duplicate log outputs in environments like pytest, FastAPI, or Streamlit.
            self._logger.propagate = False
            
            self._setup_logging(log_dir, log_file_name, max_bytes, backup_count)
            
    def _setup_logging(self, log_dir: str, log_file_name: str, max_bytes: int, backup_count: int):
        """Configures formatters and attaches handlers to the logger."""
        os.makedirs(log_dir, exist_ok=True)
        log_filepath = os.path.join(log_dir, log_file_name)

        # Added dynamic spacing (%-8s) so levels (INFO, WARNING, CRITICAL) align perfectly vertically
        logging_str = "[%(asctime)s | %(levelname)-8s | %(module)s.%(funcName)s:%(lineno)d]: %(message)s"
        formatter = logging.Formatter(logging_str, datefmt='%Y-%m-%d %H:%M:%S')

        # 1. Console Handler (Writes logs to stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

        # 2. Rotating File Handler (Replaced standard FileHandler to save disk space)
        file_handler = RotatingFileHandler(
            log_filepath, 
            mode='a', 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'  # Explicit encoding prevents crashes on Windows servers
        )
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        
        self._logger.info("Logging infrastructure initialized successfully.")
        
    @property
    def logger(self) -> logging.Logger:
        """Provides direct access to the underlying configured logging.Logger instance."""
        return self._logger


# Instantiate the default convenient wrapper for immediate project-wide imports
logger = Logger().logger