import os
from typing import Optional
import hydra

import lightning as L
import omegaconf
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload

from demos.a2c_demo.agent.actor_critic import A2CAgent
from demos.a2c_demo.model.mlp import PolicyMLP


# Simple LightningWorker
class Trainer(L.LightningWork):
    """Gym environment worker

    Args:
        agent (A2CAgent): Agent to interact with the environment
        agent_id (int): Agent id.
        agent_data_path (Path): Path to shared agent data.
        data_sizes_path (Path): PAth to shared agent sizes.
    Raises:
        NotImplementedError: If the game mode is not supported
    """

    def __init__(
        self,
        agent_cfg: omegaconf.DictConfig,
        model_cfg: omegaconf.DictConfig,
        optimizer_cfg: omegaconf.DictConfig,
        agent_id: int = 0,
        agent_state_dir: str = "./agent_state",
        **worker_kwargs
    ) -> None:
        super(Trainer, self).__init__(worker_kwargs)

        self.action_dim = None
        self._agent = None
        self._agent_cfg = agent_cfg
        self.agent_id = agent_id
        self._model_cfg = model_cfg
        self._model = None
        self._optimizer_cfg = optimizer_cfg
        self._optimizer = None

        self.episode_counter = 0

        # Path to save model state
        self.agent_state_dir = agent_state_dir
        os.makedirs(agent_state_dir, exist_ok=True)
        self.model_state_dict_path = Path(os.path.join(agent_state_dir, "model_state_dict_" + str(agent_id)))

    def run(self, signal: int, buffer: Payload):
        print("Trainer: training episode {}".format(self.episode_counter))
        buffer = buffer.value

        if self._model is None:
            self._model = hydra.utils.instantiate(self._model_cfg, input_dim=buffer.observations.shape[1], action_dim=self.action_dim)
        if self._optimizer is None:
            self._optimizer = hydra.utils.instantiate(self._optimizer_cfg, self._model.parameters())
        if self._agent is None:
            self._agent = hydra.utils.instantiate(self._agent_cfg, model=self._model, optimizer=self._optimizer)

        self._agent.replay_buffer = buffer
        print("Trainer: training agent")
        agent_loss, sum_rewards = self._agent.train_step()
        print("Trainer: Loss: {:.4f}, Sum of rewards: {:.4f}".format(agent_loss, sum_rewards))

        # Save model and optim state
        print("Trainer: saving model state dict")
        torch.save(self._model.state_dict(), self.model_state_dict_path)

        self.episode_counter += 1
