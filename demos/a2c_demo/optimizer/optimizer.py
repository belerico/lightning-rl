import os
from typing import List, Optional

import hydra
import lightning as L
import omegaconf
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload

from . import logger


class Optimizer(L.LightningWork):
    """Worker that sync the gradients received by the Trainers.

    Args:
        input_dim (int): input dimension of the model (the size of the observation space)
        action_dim (int): the action dimension of the model (the size of the action space)
        num_agents (int): the number of agents
        model_cfg (omegaconf.DictConfig): the model configuration. For this demo we have a simple linear model
            that outputs both the policy over actions and the value of the state.
        optimizer_cfg (omegaconf.DictConfig): the optimizer config.
        save_dir (Path): directory to the shared model state dict. Defaults to "./synced_model".
    """
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_agents: int,
        model_cfg: omegaconf.DictConfig,
        optimizer_cfg: omegaconf.DictConfig,
        save_dir: str = "./synced_model",
        **worker_kwargs
    ):
        super().__init__(**worker_kwargs)
        self._input_dim = input_dim
        self._action_dim = action_dim
        self.num_agents = num_agents
        self._model_cfg = model_cfg
        self._optimizer_cfg = optimizer_cfg
        self._synced_agents = 0
        self.done = False

        self._model = hydra.utils.instantiate(self._model_cfg, input_dim=self._input_dim, action_dim=self._action_dim)
        self._optimizer = hydra.utils.instantiate(self._optimizer_cfg, self._model.parameters())

        # Path to save the global model state
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model_state_dict_path = Path(os.path.join(save_dir, "model_state_dict.pt"))

    def run(self, signal: int, agent_id: int, gradients: Optional[Payload] = None, *args, **kwargs):
        if gradients is not None:
            logger.info("Optimizer: received gradients from agent {}".format(agent_id))
            self._synced_agents += 1
            gradients: List[torch.nn.Parameter] = gradients.value
            for param, grad in zip(self._model.parameters(), gradients):
                if param.grad is None:
                    param.grad = grad / self.num_agents
                else:
                    param.grad += grad / self.num_agents
        if self._synced_agents == self.num_agents:
            logger.info("Optimizer: synced all agents")
            self._optimizer.step()
            self._model.zero_grad()
            self._synced_agents = 0
            torch.save(self._model.state_dict(), self.model_state_dict_path)
            self.done = True
