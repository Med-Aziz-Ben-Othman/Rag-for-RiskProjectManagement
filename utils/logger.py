import logging
from typing import Optional, Type

from config import LOG_FORMAT

class Logger:
    _rootLogger: Optional[logging.Logger] = None

    @staticmethod
    def _setup_logger(name: str,log_level: str = "info") -> logging.Logger:
        """
        Setup and return a logger with the specified name.

        Args:
            name (str): The name of the logger.
            log_level (str): The log level for the logger. Defaults to Info

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(Logger.extract_level_from_string(log_level))

        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(LOG_FORMAT)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    @staticmethod
    def get_root_logger(name: str = None) -> logging.Logger:
        """
        Get or create the root logger with the specified name.

        Args:
            name (str): The name of the root logger.

        Returns:
            logging.Logger: Configured root logger instance.
            
        Raises:
            ValueError: If name is None when creating a new logger.
        """
        if Logger._rootLogger is None:
            if name is None:
                raise ValueError("Logger name cannot be None when creating a logger")
            Logger._rootLogger = Logger._setup_logger(name)
        return Logger._rootLogger

    @staticmethod
    def get_logger(cls: Type,log_level: str | None = None) -> logging.Logger:
        """
        Create a hierarchical logger based on the class provided.

        Args:
            cls (Type): The class for which the logger is to be created.
            log_level (str | None): The log level for the logger , Defaults to the root logger.
        Returns:
            logging.Logger: Configured hierarchical logger instance.
        """
        if Logger._rootLogger is None:
            raise ValueError("Root logger is not set. Call get_root_logger() first.")
        
        root_logger_name = Logger._rootLogger.name
        logger_name = f"{root_logger_name}.{cls.__module__}"

        logger = logging.getLogger(logger_name)
        if log_level != None:
            try:
                logger.setLevel(Logger.extract_level_from_string(log_level))
            except Exception:
                pass
        return logger

    @staticmethod
    def extract_level_from_string(log_level: str) -> int:
        """
        Extract the logging level from the string.

        Args:
            log_level (str): The log level string.

        Returns:
            int: The logging level.
        """
        return getattr(logging, log_level.upper())