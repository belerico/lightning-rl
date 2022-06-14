from typing import Optional

import torch
import torch.nn.functional as F

from lightning_rl.agent.base import Agent


class A2C(Agent):
    """Initialize the agent.

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
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        clip_gradients: Optional[float] = 0.0,
        agent_id: Optional[int] = None,
    ):
        super(A2C, self).__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=batch_size,
            clip_gradients=clip_gradients,
            agent_id=agent_id,
        )

    def train_step(self) -> None:
        """Compute the A2C loss"""

        num_samples = len(self.buffer)
        slices, idxes = self.get_batches(num_samples)

        total_value_loss = 0
        total_policy_loss = 0
        for batch_num in range(len(slices) - 1):
            batch_idxes = idxes[slices[batch_num] : slices[batch_num + 1]]
            buffer_data = self.buffer[batch_idxes]
            observation = buffer_data["observations"]
            game_actions = buffer_data["actions"]
            advantages = buffer_data["advantages"]
            returns = buffer_data["returns"]

            log_probs, values = self.evaluate_action(observation, game_actions)
            total_policy_loss -= (log_probs * advantages).sum()
            total_value_loss += F.smooth_l1_loss(values, returns, reduction="sum")

        loss = (total_policy_loss + total_value_loss) / num_samples

        self.metrics["Loss/Agent-{}/loss".format(self.agent_id)] = loss.item()
        self.metrics["Loss/Agent-{}/value_loss".format(self.agent_id)] = total_value_loss.item()
        self.metrics["Loss/Agent-{}/policy_loss".format(self.agent_id)] = total_policy_loss.item()

        self.backward(loss)
        grad_norm = self.optimize_step()
        self.metrics["Gradients/Agent-{}/grad_norm".format(self.agent_id)] = grad_norm.item()
