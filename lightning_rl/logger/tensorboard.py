from typing import Any, Dict, Optional

import lightning as L
import tensorboard
from lightning.storage.path import Path
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

    def __init__(self, log_dir: str, port: str = "6006", **work_kwargs):
        super().__init__(**work_kwargs)
        self.log_dir = Path(log_dir)
        self._tb = tensorboard.program.TensorBoard()
        if is_port_used(int(port), host=self.host):
            port = str(find_free_port(host=self.host))
        self._tb.configure(argv=[None, "--logdir", self.log_dir.name, "--host", self.host, "--port", port])
        self.url = self._tb.launch()
        self._logger = TensorBoardLogger(save_dir=self.log_dir.name, name="lightning-rl")

    def run(self, episode_counter: int, metrics: Optional[Dict[str, Any]] = None):
        if metrics is not None:
            self._logger.log_metrics(metrics, episode_counter)
