import base64
import logging
import socket
from contextlib import closing
from typing import Iterable, Union

import streamlit as st
import torch
from torch._six import inf

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


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


def compute_norm(
    parameters: _tensor_or_tensors, norm_type: float = 2.0, error_if_nonfinite: bool = False
) -> torch.Tensor:
    r"""Compute tensor norm of an iterable of tensors.

    The norm is computed over all tensors together, as if they were
    concatenated into a single vector.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor to be nornmalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the tensors from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]), norm_type
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm
