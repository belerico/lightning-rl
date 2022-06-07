import os
from typing import List, Optional

import hydra
import lightning as L
import numpy as np
import omegaconf
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload

from demos.a2c_demo.buffer.rollout import RolloutBuffer

from . import logger


class Trainer(L.LightningWork):
    """Worker that train the agent given the observations received by the Players.

    Args:
        input_dim (int): input dimension of the model (the size of the observation space)
        action_dim (int): the action dimension of the model (the size of the action space)
        num_players (int): the number of players in the game.
        agent_cfg (omegaconf.DictConfig): the agent configuration. The agent specifies the reinforcement learning
            algorithm to use. For this demo, we use the A2C algorithm (https://arxiv.org/abs/1602.01783).
        model_cfg (omegaconf.DictConfig): the model configuration. For this demo we have a simple linear model
            that outputs both the policy over actions and the value of the state.
        optimizer_cfg (omegaconf.DictConfig): the optimizer configuration. For this demo we use the Adam optimizer by default.
        model_state_dict_path (str): the path to the model state dict. Default is "./synced_model/model_state_dict.pth".
        agent_id (int, optional): the agent id. Defaults to 0.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_players: int,
        agent_cfg: omegaconf.DictConfig,
        model_cfg: omegaconf.DictConfig,
        optimizer_cfg: omegaconf.DictConfig,
        model_state_dict_path: str = "./synced_model/model_state_dict.pth",
        agent_id: int = 0,
        **worker_kwargs
    ) -> None:
        super(Trainer, self).__init__(worker_kwargs)
        self.agent_id = agent_id
        self.num_players = num_players
        self._buffer: RolloutBuffer = None
        model = hydra.utils.instantiate(model_cfg, input_dim=input_dim, action_dim=action_dim)
        optimizer = hydra.utils.instantiate(optimizer_cfg, model.parameters())
        self._agent = hydra.utils.instantiate(agent_cfg, agent_id=self.agent_id, model=model, optimizer=optimizer)
        self.episode_counter = 0
        os.makedirs(os.path.dirname(model_state_dict_path), exist_ok=True)
        self.model_state_dict_path = Path(model_state_dict_path)
        self.metrics = None
        self.has_done = False

    def run(self, signal: int, buffers: Optional[List[Payload]] = None):
        if signal > 0 and not any(buffer is None for buffer in buffers):
            logger.info("Trainer-{}: received buffer from Players".format(self.agent_id))
            self._buffer = buffers[0].value
            for i in range(1, self.num_players):
                self._buffer.append(buffers[i].value)
            logger.info(
                "Trainer-{}: training episode {}, buffer length: {}".format(
                    self.agent_id, self.episode_counter, len(self._buffer)
                )
            )
            sum_rewards = np.sum(self._buffer.rewards).item() / self.num_players
            self._agent.buffer = self._buffer
            metrics = self._agent.train_step()
            torch.save(self._agent.model.state_dict(), self.model_state_dict_path)
            metrics["Game/Agent-{}/episode_length".format(self.agent_id)] = len(self._buffer) / self.num_players
            metrics["Rewards/Agent-{}/sum_rew".format(self.agent_id)] = sum_rewards
            logger.info(
                "Trainer-{}: Loss: {:.4f}, Policy Loss: {:.4f}, Value Loss: {:.4f}, Sum of rewards: {:.4f}".format(
                    self.agent_id,
                    metrics["Loss/Agent-{}/loss".format(self.agent_id)],
                    metrics["Loss/Agent-{}/policy_loss".format(self.agent_id)],
                    metrics["Loss/Agent-{}/value_loss".format(self.agent_id)],
                    sum_rewards,
                )
            )
            self._buffer = None
            self.metrics = metrics
            self.episode_counter += 1
            self.has_done = True
