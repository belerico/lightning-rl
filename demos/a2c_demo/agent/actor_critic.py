from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from demos.a2c_demo.buffer.rollout import RolloutBuffer


class A2CAgent:
    """Initialize the agent.

    Args:
        model (torch.nn.Module): model of the Neural Net for the Actor and the Critic.
        optimizer (torch.optim.Optimizer, optional): optimizer for performing the parameters update step after the backward.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): scheduler for the learning rate decay.
            Default is None.
        batch_size (int, optional): size for the minibatch. Default is 32.
        clip_gradients (float, optional): clip parameter for .nn.utils.clip_grad_norm_. Does not clip if the value
            is None or smaller than 0. Default is 0.0.
        entropy_coeff (float, optional): coefficient for the entropy regularization. Default is None.
        normalize_returns (bool, optional): whether to normalize the returns.
        agent_id (int, optional): The agent id.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        clip_gradients: Optional[float] = 0.0,
        entropy_coeff: Optional[float] = None,
        normalize_returns: bool = True,
        agent_id: Optional[int] = None,
    ):
        super(A2CAgent, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distribution = torch.distributions.Categorical
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.clip_gradients = clip_gradients
        self.episode_counter = 0
        self.returns: Optional[torch.Tensor] = None
        self._buffer = None
        self.agent_id = agent_id
        self.metrics = {}
        self.normalize_returns = normalize_returns

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

    def evaluate_action(
        self, observation: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Select an action for the given observation and optional state.

        Args:
            observation (torch.Tensor): The observation to select an action for.
            actions (torch.Tensor): Actions to be evaluated.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor] torch.Tensor]: Returns a tuple containing the actions log-probabilities,
            optionally it returns the entropy of the policy and the critic values.
        """
        action_probs, value = self.model(observation)
        dist = self.distribution(probs=action_probs)
        log_probs = dist.log_prob(actions.squeeze(1)).unsqueeze(1)
        entropy = None
        if self.entropy_coeff is not None:
            entropy = dist.entropy()
        return log_probs, entropy, value

    @torch.no_grad()
    def optimize_step(self) -> Optional[torch.Tensor]:
        """Optimize the agent's parameters.

        Returns:
            Optional[torch.Tensor]: the norm of the gradients before clipping it.
        """
        grad_norm = None
        if self.clip_gradients is not None and self.clip_gradients > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradients)
        if self.optimizer is not None:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return grad_norm

    def before_backward(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def after_backward(self):
        pass

    def backward(self, loss: torch.Tensor):
        self.before_backward()
        loss.backward()
        self.after_backward()

    def get_slices_and_indexes(self, num_samples: int) -> Tuple[List[int], List[int]]:
        slices = list(range(0, num_samples + 1, self.batch_size))
        if num_samples % self.batch_size != 0:
            slices += [num_samples]
        elif len(slices) == 1 and slices[0] == 0:
            slices.append(num_samples)
        idxes = list(range(num_samples))
        return slices, idxes

    def compute_loss(self) -> None:
        """Compute the A2C loss"""

        # Get slices and indexes for batching
        num_samples = len(self.buffer)
        slices, idxes = self.get_slices_and_indexes(num_samples)

        total_value_loss = 0
        total_policy_loss = 0
        total_entropy_loss = 0
        for batch_num in range(len(slices) - 1):
            batch_idxes = idxes[slices[batch_num] : slices[batch_num + 1]]
            buffer_data = self.buffer[batch_idxes]
            observation = buffer_data["observations"]
            game_actions = buffer_data["actions"]
            advantages = buffer_data["advantages"]
            returns = buffer_data["returns"]

            log_probs, entropy, values = self.evaluate_action(observation, game_actions)
            total_policy_loss -= (log_probs * advantages).sum()
            total_value_loss += F.smooth_l1_loss(values, returns, reduction="sum")
            if entropy is not None:
                total_entropy_loss -= self.entropy_coeff * entropy.sum()

        loss = (total_policy_loss + total_value_loss + total_entropy_loss) / num_samples

        self.metrics["Loss/Agent-{}/loss".format(self.agent_id)] = loss.item()
        self.metrics["Loss/Agent-{}/value_loss".format(self.agent_id)] = total_value_loss.item()
        self.metrics["Loss/Agent-{}/policy_loss".format(self.agent_id)] = total_policy_loss.item()

        self.backward(loss)
        grad_norm = self.optimize_step()
        self.metrics["Gradients/Agent-{}/grad_norm".format(self.agent_id)] = grad_norm.item()

    def train_step(self) -> Tuple[List[torch.nn.Parameter], Dict[str, Any]]:
        """Run the forward and backward passes."""
        self.metrics = {}
        self.compute_loss()
        return self.metrics
