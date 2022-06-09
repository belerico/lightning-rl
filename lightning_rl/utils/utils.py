import logging


def get_logger(
    name: str,
    level: int = logging.INFO,
    message_fmt: str = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
) -> logging.Logger:
    formatter = logging.Formatter(message_fmt, datefmt=None, style="%")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
