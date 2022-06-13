import subprocess
from typing import Any, Dict, Optional

import lightning as L
from lightning.storage import Drive
from pytorch_lightning.loggers import TensorBoardLogger


class TensorboardWork(L.LightningWork):
    """Tensorboard logger as a LightningWork

    Args:
        local_log_dir (str): where to save logs.
        host (str): host to run tensorboard.
        port (str): port to run tensorboard.
        **work_kwargs: additional arguments to pass to LightningWork.
    """

    def __init__(self, local_log_dir: str = "logs", port: str = "6006", **work_kwargs):
        super().__init__(**work_kwargs)
        self.local_log_dir = local_log_dir
        self._logger = TensorBoardLogger(self.local_log_dir)
        self._tensorboard_started = False
        self._local_log_dir_added = False

    @property
    def tensorboard_url(self) -> str:
        return self.url

    @property
    def tensorboard_log_dir(self) -> str:
        return self._logger.log_dir

    def run(self, episode_counter: int, metrics: Optional[Dict[str, Any]] = None, drive: Optional[Drive] = None):
        if not self._tensorboard_started:
            subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir",
                    self.local_log_dir,
                    "--host",
                    self.host,
                    "--port",
                    str(self.port),
                ]
            )
            self._tensorboard_started = True
        if metrics is not None:
            self._logger.log_metrics(metrics, episode_counter)
            if not self._local_log_dir_added and drive is not None:
                drive.put(self.local_log_dir)
                self._local_log_dir_added = True
