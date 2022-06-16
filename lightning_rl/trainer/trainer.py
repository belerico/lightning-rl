import os
from typing import List, Optional

import hydra
import lightning as L
import numpy as np
import omegaconf
import torch
from lightning.app.storage.payload import Payload

from lightning_rl.buffer.rollout import RolloutBuffer

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
        checkpoint_path (str, optional): the path to the model state dict. Default is "./checkpoints/checkpoint.pth".
        max_buffer_length (int, optional): the maximum length of the buffer. If the `max_buffer_length` is > 0, then the trainer will
            start training if its buffer has at least `max_buffer_size` elements (This is useful if one runs the PPO agent).
            If `max_buffer_length` is <= 0, then no concatenation will be done and the trainer will train as soon as every player
            has done playing (this is useful if one runs the A2C agent).
            Default is -1.
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
        checkpoint_path: str = "./checkpoints/checkpoint.pth",
        max_buffer_length: int = -1,
        agent_id: int = 0,
        **work_kwargs
    ) -> None:
        super(Trainer, self).__init__(work_kwargs)
        if "PPO" in agent_cfg._target_ and max_buffer_length <= 0:
            logger.warn("Trainer-{}: PPO requires a buffer length > 0".format(agent_id))
        self.agent_id = agent_id
        self._num_players = num_players
        self._buffer: RolloutBuffer = None
        model = hydra.utils.instantiate(model_cfg, input_dim=input_dim, action_dim=action_dim)
        optimizer = hydra.utils.instantiate(optimizer_cfg, model.parameters())
        self._agent = hydra.utils.instantiate(agent_cfg, agent_id=self.agent_id, model=model, optimizer=optimizer)
        self.episode_counter = 0
        self.checkpoint_path = None
        self.local_checkpoint_path = checkpoint_path
        self._checkpoint_path_name, self._checkpoint_path_ext = os.path.splitext(checkpoint_path)
        self._max_buffer_length = max_buffer_length
        self.metrics = None
        self.first_time_model_save = False
        self._episodes_delta = 0

    def run(self, signal: int, buffers: Optional[List[Payload]] = None):
        if self.checkpoint_path is None:
            self.checkpoint_path = "lit://" + self.local_checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        if signal > 0 and not any(buffer is None for buffer in buffers):
            self._episodes_delta += 1
            logger.info("Trainer-{}: received buffer from Players".format(self.agent_id))
            start_idx = 0
            if self._buffer is None:
                self._buffer = buffers[0].value
                start_idx = 1
            for i in range(start_idx, self._num_players):
                self._buffer.append(buffers[i].value)
            logger.info("Trainer-{}: buffer size: {}".format(self.agent_id, len(self._buffer)))
            if (
                self._buffer is not None and len(self._buffer) >= self._max_buffer_length
            ) or self._max_buffer_length <= 0:
                logger.info(
                    "Trainer-{}: training episode {}, buffer length: {}".format(
                        self.agent_id, self.episode_counter, len(self._buffer)
                    )
                )
                if self._max_buffer_length > 0:
                    self._buffer.shrink(self._max_buffer_length)
                sum_rewards = np.sum(self._buffer.rewards).item() / self._num_players / self._episodes_delta
                self._agent.buffer = self._buffer
                metrics = self._agent.train()
                metrics["Game/Agent-{}/episode_length".format(self.agent_id)] = (
                    len(self._buffer) / self._num_players / self._episodes_delta
                )
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
                model_state_dict = self._agent.model.state_dict()
                torch.save(model_state_dict, self.checkpoint_path)
                self._buffer = None
                self.metrics = metrics
                self._episodes_delta = 0
            self.episode_counter += 1
        elif signal == 0:
            torch.save(self._agent.model.state_dict(), self.checkpoint_path)
            self.first_time_model_save = True
