import os
import subprocess
from typing import Any, Dict, Optional

import lightning as L
from lightning.storage import Drive
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_rl.utils.utils import find_free_port, is_port_used


class TensorboardWork(L.LightningWork):
    """Tensorboard logger as a LightningWork

    Args:
        log_dir (str): where to save logs.
        host (str): host to run tensorboard.
        port (str): port to run tensorboard.
        **work_kwargs: additional arguments to pass to LightningWork.
    """

    def __init__(self, port: str = "6006", **work_kwargs):
        super().__init__(**work_kwargs)
        self.log_dir = Drive("lit://logs")
        if is_port_used(int(port), host=self.host):
            port = str(find_free_port(host=self.host))
        self._logger = TensorBoardLogger(str(self.log_dir))
        self.tensorboard_started = False

    @property
    def tensorboard_url(self) -> str:
        return self.url

    def run(self, episode_counter: int, metrics: Optional[Dict[str, Any]] = None):
        if not self.tensorboard_started:
            subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir",
                    str(self.log_dir),
                    "--host",
                    self.host,
                    "--port",
                    str(self.port),
                ]
            )
            self.tensorboard_started = True
        if metrics is not None:
            self._logger.log_metrics(metrics, episode_counter)
