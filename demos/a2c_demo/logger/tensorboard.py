from typing import Any, Dict

import lightning as L
import tensorboard
from lightning.storage.path import Path
from pytorch_lightning.loggers import TensorBoardLogger


class TensorboardWork(L.LightningWork):
    def __init__(self, log_dir: str, num_agents: int, **worker_kwargs):
        super().__init__(**worker_kwargs)
        self.log_dir = Path(log_dir)
        self._tb = tensorboard.program.TensorBoard()
        self._tb.configure(argv=[None, "--logdir", self.log_dir.name])
        self._tb.launch()
        self._logger = TensorBoardLogger(save_dir=self.log_dir.name, name="a2c_demo")
        self._metrics = {}
        self._num_agents = num_agents
        self._metrics_received = 0

    def run(self, signal: int, metrics: Dict[str, Any]):
        if metrics is not None:
            self._metrics_received += 1
            if self._metrics_received == 0:
                self._metrics = metrics
            else:
                self._metrics.update(metrics)
        if self._metrics_received == self._num_agents:
            self._logger.log_metrics(self._metrics, signal)
            self._metrics = {}
            self._metrics_received = 0
