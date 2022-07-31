from typing import Any, Dict, List, Optional, Tuple

import hydra
import omegaconf
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from lightning_rl.buffer.rollout import RolloutBuffer
from lightning_rl.utils.utils import compute_norm


class Agent:
    """A generic RL agent.

    Args:
        model (torch.nn.Module): model of the Neural Net for the Actor and the Critic.
        optimizer (torch.optim.Optimizer, optional): optimizer for performing the parameters update step after the backward.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): scheduler for the learning rate decay.
            Default is None.
        batch_size (int, optional): size for the minibatch. Default is 32.
        clip_gradients (float, optional): clip parameter for .nn.utils.clip_grad_norm_. Does not clip if the value
            is None or smaller than 0. Default is 0.0.
        agent_id (int, optional): The agent id.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        model_cfg: omegaconf.DictConfig,
        optimizer_cfg: omegaconf.DictConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        clip_gradients: Optional[float] = 0.0,
        agent_id: Optional[int] = None,
        distributed: bool = False,
    ):
        super(Agent, self).__init__()
        model = hydra.utils.instantiate(model_cfg, input_dim=input_dim, action_dim=action_dim)
        self.optimizer = hydra.utils.instantiate(optimizer_cfg, model.parameters())
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            self.model = DistributedDataParallel(model)
        else:
            self.model = model
        self.scheduler = scheduler
        self.distribution = torch.distributions.Categorical
        self.batch_size = batch_size
        self.clip_gradients = clip_gradients
        self._buffer = None
        self.agent_id = agent_id
        self.distributed = distributed
        self.metrics = {}

    @property
    def buffer(self):
        """Get the replay buffer.

        Returns:
            buffer (RolloutBuffer): the replay buffer.
        """
        return self._buffer

    @buffer.setter
    def buffer(self, buffer: RolloutBuffer):
        """Set the replay buffer.

        Args:
            buffer (RolloutBuffer): the replay buffer.
        """
        self._buffer = buffer

    def select_greedy_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Select an action with greedy strategy. It picks the best action according to the current policy probabilities for the given observation.

        Args:
            observation (torch.Tensor): The observation to select an action for.

        Returns:
            torch.Tensor: The selected actions.
        """
        action_probs, _ = self.model(observation)
        selected_actions = torch.argmax(action_probs, dim=1)

        return selected_actions

    def select_action(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action for the given observation and optional state.

        Args:
            observation (torch.Tensor): The observation to select an action for.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns a tuple containing the selected actions,
            the corresponding log-probabilities and the critic values.
        """
        action_probs, value = self.model(observation)
        dist = self.distribution(probs=action_probs)
        selected_actions = dist.sample()
        log_probs = dist.log_prob(selected_actions)
        return selected_actions, log_probs, value

    def evaluate_action(self, observation: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select an action for the given observation and optional state.

        Args:
            observation (torch.Tensor): The observation to select an action for.
            actions (torch.Tensor): Actions to be evaluated.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor] torch.Tensor]: Returns a tuple containing the actions log-probabilities and the critic values.
        """
        action_probs, value = self.model(observation)
        dist = self.distribution(probs=action_probs)
        log_probs = dist.log_prob(actions.squeeze(1)).unsqueeze(1)
        return log_probs, value

    @torch.no_grad()
    def optimize_step(self) -> Optional[torch.Tensor]:
        """Optimize the agent's parameters.

        Returns:
            Optional[torch.Tensor]: the norm of the gradients before clipping it.
        """
        if self.clip_gradients is not None and self.clip_gradients > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradients)
        else:
            grad_norm = compute_norm(self.model.parameters())
        if self.optimizer is not None:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return grad_norm

    def backward(self, loss: torch.Tensor):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

    def get_batches(self, num_samples: int) -> Tuple[List[int], List[int]]:
        slices = list(range(0, num_samples + 1, self.batch_size))
        if num_samples % self.batch_size != 0:
            slices += [num_samples]
        elif len(slices) == 1 and slices[0] == 0:
            slices.append(num_samples)
        idxes = list(range(num_samples))
        return slices, idxes

    def train_step(self) -> None:
        raise NotImplementedError

    def train(self) -> Dict[str, Any]:
        """Run the forward and backward passes.

        Returns:
            Dict[str, Any]: the collected metrics. The metrics are collected in the following way:
                - Loss/Agent-{agent_id}/loss: the loss.
                - Loss/Agent-{agent_id}/value_loss: the value loss.
                - Loss/Agent-{agent_id}/policy_loss: the policy loss.
                - Gradients/Agent-{agent_id}/grad_norm: the norm of the gradients.
        """
        self.metrics = {}
        self.train_step()
        return self.metrics
