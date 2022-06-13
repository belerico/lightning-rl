import base64
import logging
import socket
from contextlib import closing

import streamlit as st


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


def logo_and_title(logo_path: str, obj=None):
    if obj is None:
        obj = st
    obj.markdown(
        """
        <style>
        .logo-text {
            font-weight: 700;
            font-size: 50px;
            display:inline-block;
            vertical-align:middle;
        }
        .logo-img {
            height: 50px;
            padding-right: 10px;
            width: auto;
            vertical-align:middle;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    obj.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}">
            <div class="logo-text">Lightning RL Demo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def find_free_port(host: str = "localhost") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def is_port_used(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
