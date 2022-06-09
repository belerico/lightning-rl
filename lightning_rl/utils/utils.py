import logging
import socket
from contextlib import closing


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


def find_free_port(host: str = "localhost") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def is_port_used(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
