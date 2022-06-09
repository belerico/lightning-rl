from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchmetrics import MeanMetric

from lightning_rl.agent.base import Agent


class PPOAgent(Agent):
    """Initialize the agent.

    Args:
        model (torch.nn.Module): model of the Neural Net for the Actor and the Critic.
        optimizer (torch.optim.Optimizer, optional): optimizer for performing the parameters update step after the backward.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): scheduler for the learning rate decay.
            Default is None.
        clip_coeff (float, optional): coefficient for the PPO surrogate loss. Default is 0.2.
        epochs (int, optional): number of epochs for the training. Default is 10.
        batch_size (int, optional): size for the minibatch. Default is 32.
        shuffle (bool, optional): whether to shuffle the data. Default is True.
        clip_gradients (float, optional): clip parameter for .nn.utils.clip_grad_norm_. Does not clip if the value
            is None or smaller than 0. Default is 0.0.
        entropy_coeff (float, optional): coefficient for the entropy regularization. Default is None.
        agent_id (int, optional): The agent id.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        clip_coeff: float = 0.2,
        epochs: int = 10,
        batch_size: int = 32,
        shuffle: bool = True,
        clip_gradients: Optional[float] = 0.0,
        entropy_coeff: Optional[float] = None,
        agent_id: Optional[int] = None,
    ):
        super(PPOAgent, self).__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=batch_size,
            clip_gradients=clip_gradients,
            entropy_coeff=entropy_coeff,
            agent_id=agent_id,
        )
        self.clip_coeff = clip_coeff
        self.epochs = epochs
        self.shuffle = shuffle
        self.total_loss = MeanMetric()
        self.total_policy_loss = MeanMetric()
        self.total_value_loss = MeanMetric()
        self.total_grads_norm = MeanMetric()

    def compute_loss(self) -> None:
        if self.shuffle:
            sampler = RandomSampler(self.buffer)
        else:
            sampler = SequentialSampler(self.buffer)
        slices, _ = self.get_slices_and_indexes(len(sampler))
        for epoch in range(self.epochs):
            idxes = list(iter(sampler))
            for batch_num in range(len(slices) - 1):
                batch_idxes = idxes[slices[batch_num] : slices[batch_num + 1]]
                buffer_data = self.buffer[batch_idxes]
                game_actions = buffer_data["actions"]
                game_log_probs = buffer_data["log_probs"]
                observations = buffer_data["observations"]
                advantages = buffer_data["advantages"].float()
                returns = buffer_data["returns"].float()
                old_log_probs = game_log_probs.sum(dim=1, keepdim=True)

                log_probs, entropy, values = self.evaluate_action(observations, game_actions)

                # PPO clipping
                ratio = torch.exp(log_probs - old_log_probs)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                self.total_policy_loss.update(policy_loss)

                # Value loss
                value_loss = F.mse_loss(values, returns)
                self.total_value_loss.update(value_loss)

                loss = policy_loss + value_loss
                self.total_loss.update(loss)

                # Backward pass
                self.backward(loss)

                # Clip gradients + optimizer step
                grad_norm = self.optimize_step()
                self.total_grads_norm.update(grad_norm)

        self.metrics["Loss/Agent-{}/loss".format(self.agent_id)] = self.total_loss.compute().item()
        self.metrics["Loss/Agent-{}/value_loss".format(self.agent_id)] = self.total_value_loss.compute().item()
        self.metrics["Loss/Agent-{}/policy_loss".format(self.agent_id)] = self.total_policy_loss.compute().item()
        self.metrics["Gradients/Agent-{}/grad_norm".format(self.agent_id)] = self.total_grads_norm.compute().item()
