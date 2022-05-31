
import hydra
import lightning as L
import numpy as np
import omegaconf
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload

from . import logger


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
        model_state_dict_path: Path,
        agent_id: int = 0,
        **worker_kwargs
    ) -> None:
        super(Trainer, self).__init__(worker_kwargs)
        self._input_dim = input_dim
        self._action_dim = action_dim
        self.agent_id = agent_id
        model = hydra.utils.instantiate(model_cfg, input_dim=self._input_dim, action_dim=self._action_dim)
        self._agent = hydra.utils.instantiate(agent_cfg, agent_id=self.agent_id, model=model, optimizer=None)
        self.gradients = None  # Payload to be shared
        self.episode_counter = 0
        self.model_state_dict_path = model_state_dict_path
        self.metrics = None

    def run(self, signal: int, buffer: Payload):
        if signal > 0:
            logger.info("Trainer-{}: training episode {}".format(self.agent_id, self.episode_counter))
            buffer = buffer.value
            sum_rewards = np.sum(buffer.rewards).item()

            if self.model_state_dict_path.exists():
                self._agent.model.load_state_dict(torch.load(self.model_state_dict_path))

            self._agent.replay_buffer = buffer
            gradients, metrics = self._agent.train_step()
            metrics["Game/Agent-{}/episode_length".format(self.agent_id)] = len(buffer)
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
            self.metrics = metrics
            self.gradients = Payload(gradients)
            self.episode_counter += 1
