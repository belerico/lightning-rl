import os
from typing import List, Optional

import hydra
import lightning as L
import numpy as np
import omegaconf
import torch
import torch.distributed
from lightning.app.storage.payload import Payload
from torch.nn.parallel import DistributedDataParallel

from lightning_rl.agent.base import Agent
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
        **work_kwargs,
    ) -> None:
        super(Trainer, self).__init__(work_kwargs)
        if "PPO" in agent_cfg._target_ and max_buffer_length <= 0:
            logger.warn("Trainer-{}: PPO requires a buffer length > 0".format(agent_id))

        # Private attributes
        self._input_dim = input_dim
        self._action_dim = action_dim
        self._episodes_delta = 0
        self._num_players = num_players
        self._buffer: RolloutBuffer = None
        self._agent_cfg = agent_cfg
        self._model_cfg = model_cfg
        self._optimizer_cfg = optimizer_cfg
        self._agent: Agent = None
        self._checkpoint_path_name, self._checkpoint_path_ext = os.path.splitext(checkpoint_path)
        self._max_buffer_length = max_buffer_length

        # Public attributes
        self.agent_id = agent_id
        self.episode_counter = 0
        self.checkpoint_path = None
        self.local_checkpoint_path = checkpoint_path
        self.metrics = None
        self.first_time_model_save = False
        self.dist_initialized = False

    def run(
        self,
        signal: int,
        buffers: Optional[List[Payload]] = None,
        main_address: str = "localhost",
        main_port: int = 1111,
        world_size: str = 1,
        rank: str = 0,
        rank_zero_init: bool = False,
        init_process_group: bool = False,
    ):
        # Initialize DDP process group
        if rank_zero_init:
            return

        if init_process_group and not torch.distributed.is_initialized():
            logger.info(f"Initializing process group: {main_address=}, {main_port=}, {world_size=}, {rank=}")
            os.environ["MASTER_ADDR"] = main_address
            os.environ["MASTER_PORT"] = str(main_port)
            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            torch.distributed.init_process_group(
                backend="gloo", init_method=f"tcp://{main_address}:{main_port}", world_size=world_size, rank=rank
            )
            logger.info(f"Finished initializing process group")
            self.dist_initialized = True

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Init the agent here so that DDP is already initialized
            if self._agent is None:
                self._agent = hydra.utils.instantiate(
                    self._agent_cfg,
                    input_dim=self._input_dim,
                    action_dim=self._action_dim,
                    model_cfg=self._model_cfg,
                    optimizer_cfg=self._optimizer_cfg,
                    agent_id=self.agent_id,
                    distributed=True,
                    _recursive_=False
                )

            # Trainer logic
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
                for i in range(start_idx, len(buffers)):
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
                    sum_rewards = np.sum(self._buffer.rewards).item() / len(buffers) / self._episodes_delta
                    self._agent.buffer = self._buffer
                    metrics = self._agent.train()
                    metrics["Game/Agent-{}/episode_length".format(self.agent_id)] = (
                        len(self._buffer) / len(buffers) / self._episodes_delta
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
                    if self.agent_id == 0:
                        if isinstance(self._agent.model, DistributedDataParallel):
                            model_state_dict = self._agent.model.module.state_dict()
                        else:
                            model_state_dict = self._agent.model.state_dict()
                        torch.save(model_state_dict, self.checkpoint_path)
                    self._buffer = None
                    self.metrics = metrics
                    self._episodes_delta = 0
                self.episode_counter += 1
            elif signal == 0:
                if isinstance(self._agent.model, DistributedDataParallel):
                    model_state_dict = self._agent.model.module.state_dict()
                else:
                    model_state_dict = self._agent.model.state_dict()
                torch.save(model_state_dict, self.checkpoint_path)
                self.first_time_model_save = True
