import os

import hydra
import lightning as L
import omegaconf
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload


# Simple LightningWorker
class Trainer(L.LightningWork):
    """Worker that wraps a gym environment and plays in it.

    Args:
        input_dim (int): input dimension of the model (the size of the observation space)
        action_dim (int): the action dimension of the model (the size of the action space)
        agent_cfg (omegaconf.DictConfig): the agent configuration. The agent specifies the reinforcement learning
            algorithm to use. For this demo, we use the A2C algorithm (https://arxiv.org/abs/1602.01783).
        model_cfg (omegaconf.DictConfig): the model configuration. For this demo we have a simple linear model
            that outputs both the policy over actions and the value of the state.
        optimizer_cfg (omegaconf.DictConfig): the optimizer configuration. By default the Adam optimizer is used.
        model_state_dict_path (Path): shared path to the model state dict.
        agent_id (int, optional): the agent id. Defaults to 0.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        agent_cfg: omegaconf.DictConfig,
        model_cfg: omegaconf.DictConfig,
        optimizer_cfg: omegaconf.DictConfig,
        model_state_dict_path: Path,
        agent_id: int = 0,
        **worker_kwargs
    ) -> None: 
        super(Trainer, self).__init__(worker_kwargs)
        self._input_dim = input_dim
        self._action_dim = action_dim
        self._agent = None
        self._agent_cfg = agent_cfg
        self.agent_id = agent_id
        self._model_cfg = model_cfg
        self._optimizer_cfg = optimizer_cfg
        self.gradients = None  # Payload to be shared
        self.episode_counter = 0
        self.model_state_dict_path = model_state_dict_path

    def run(self, signal: int, buffer: Payload):
        if signal > 0:
            print("Trainer: training episode {}".format(self.episode_counter))
            buffer.get()
            buffer = buffer.value

            if self._agent is None:
                model = hydra.utils.instantiate(self._model_cfg, input_dim=self._input_dim, action_dim=self._action_dim)
                optimizer = hydra.utils.instantiate(self._optimizer_cfg, model.parameters())
                self._agent = hydra.utils.instantiate(self._agent_cfg, model=model, optimizer=optimizer)

            if self.model_state_dict_path.exists():
                print("Trainer: loading synced gradients")
                self._agent.model.load_state_dict(torch.load(self.model_state_dict_path))

            self._agent.replay_buffer = buffer
            print("Trainer: training agent")
            agent_loss, sum_rewards, gradients = self._agent.train_step()
            print("Trainer: Loss: {:.4f}, Sum of rewards: {:.4f}".format(agent_loss, sum_rewards))
            self.gradients = Payload(gradients)
            self.episode_counter += 1
