from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from demos.a2c_demo.buffer.rollout import RolloutBuffer


# TODO: wrap the agent into a LightningModule
class A2CAgent:
    """Initialize the agent.

    Args:
        model (torch.nn.Module): model of the Neural Net for the Actor and the Critic.
        optimizer (torch.optim.Optimizer): optimized for performing the parameters update step after the backward.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): scheduler for the learning rate decay.
            Default is None.
        batch_size (int, optional): size for the minibatch. Default is 32.
        shuffle (bool, optional): to shuffle or not the minibatch samples. Default is False.
        clip_gradients (float, optional): clip parameter for .nn.utils.clip_grad_norm_. Does not clip if the value
            is None or smaller than 0. Default is 0.0.
        rewards_gamma (np.ndarray, optional): discount factor parameter. Default is np.array([0.99]).
        normalize_returns (bool): whether to normalize the returns.
        normalize_advantages (bool): whether to normalize the advantages.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        shuffle: bool = False,
        clip_gradients: Optional[float] = 0.0,
        rewards_gamma: np.ndarray = np.array([0.99]),
        normalize_returns: bool = True,
        normalize_advantages: bool = True,
    ):
        super(A2CAgent, self).__init__()

        # Model
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distribution = torch.distributions.Categorical

        # Training
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.clip_gradients = clip_gradients
        self.episode_counter = 0
        self.returns: Optional[torch.Tensor] = None
        self._replay_buffer = None

        # Rewards
        self.rewards_gamma = rewards_gamma
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages

        # Checkpointing
        # TODO: add checkpointing for the model

    @property
    def replay_buffer(self):
        """Get the replay buffer.

        Returns:
            draive.buffer.RolloutBuffer: the replay buffer.
        """
        return self._replay_buffer

    @replay_buffer.setter
    def replay_buffer(self, replay_buffer: RolloutBuffer):
        """Set the replay buffer.

        Args:
            replay_buffer (RolloutBuffer): the replay buffer.
        """
        self._replay_buffer = replay_buffer

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
            state (Optional[tuple], optional): The lstm state to select an action for. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns a tuple containing the selected actions,
            the corresponding log-probabilities, the critic values.
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
            Tuple[torch.Tensor, torch.Tensor]: Returns a tuple containing the `actions` log-probabilities and the critic values.
        """
        action_probs, value = self.model(observation)
        dist = self.distribution(probs=action_probs)
        log_probs = dist.log_prob(actions)
        return log_probs, value

    def compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns

        Args:
            rewards (np.ndarray): array of rewards of every time step. It has a shape of
                TxNr, where `T` is the number of time steps, while `Nr` is the number of
                rewards.

        Returns:
        np.ndarray: the array of discounted rewards of shape Tx1 if `reduction` is "sum" or "mean",
        of shape TxNr otherwise.
        """
        assert rewards.shape[0] == len(
            self.rewards_gamma
        ), "The number of gammas {} must be equal to the number of rewards {} for every step".format(
            len(self.rewards_gamma), rewards.shape[0]
        )
        returns = np.zeros(rewards.shape)
        R = np.zeros(self.rewards_gamma.shape)
        n_steps = rewards.shape[1]
        for i, r in enumerate(rewards[:, ::-1].T):
            R = r + self.rewards_gamma * R
            returns[:, n_steps - i - 1] = R
        returns = np.sum(returns, axis=0)
        if self.normalize_returns:
            returns = (returns - np.mean(returns)) / (np.std(returns, ddof=1) + 1e-8)
        return returns

    @torch.no_grad()
    def optimize_step(self) -> Optional[torch.Tensor]:
        """Optimize the agent's parameters.

        Returns:
            Optional[torch.Tensor]: the norm of the gradients before clipping it.
        """
        # Clip gradients
        grad_norm = None
        if self.clip_gradients is not None and self.clip_gradients > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradients)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return grad_norm

    # TODO: wrap the agent into a LightningModule
    def compute_loss(self) -> torch.Tensor:
        """Compute the loss.

        Returns:
            torch.Tensor: Returns loss.
        """
        # Get returns
        returns = self.compute_returns(self.replay_buffer.rewards)
        returns = torch.from_numpy(returns)

        num_samples = len(returns)
        slices = list(range(0, num_samples + 1, self.batch_size))
        if num_samples % self.batch_size != 0:
            slices += [num_samples]
        elif len(slices) == 1 and slices[0] == 0:
            slices.append(num_samples)
        idxes = list(range(num_samples))
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(idxes)

        # Train loop
        total_value_loss = 0
        total_policy_loss = 0
        for batch_num in range(len(slices) - 1):
            batch_idxes = idxes[slices[batch_num] : slices[batch_num + 1]]
            observation, _, _, game_actions, _ = self.replay_buffer[batch_idxes]
            for game_action_idx, game_action in enumerate(game_actions):
                game_actions[game_action_idx] = game_action

            _, agent_action_dists, critic_value, next_state = self.select_action(observation, next_state)

            ret = returns[batch_idxes]
            advantage = ret - critic_value.detach()

            policy_loss = 0
            for i in range(len(game_actions)):
                policy_loss -= agent_action_dists[i].log_prob(game_actions[i]) * advantage
            total_policy_loss += policy_loss.sum() / num_samples
            total_value_loss += F.smooth_l1_loss(critic_value, ret, reduction="sum") / num_samples
        loss = total_policy_loss + total_value_loss
        return loss

    def read_agent_data(self):
        # Read agent data from Path (shared) object
        pass

    def train_step(self) -> None:
        """Run the forward and backward passes."""

        # TODO: Read agent data and set the replay buffer
        # TODO: Read `data_size` from state propagation
        agent_data = self.read_agent_data()

        # TODO: Read `block_sizes` from state propagation
        # TODO: Read `steps_num` from state propagation
        steps_num = 0
        block_sizes = []
        self.replay_buffer = RolloutBuffer.from_array(agent_data, block_sizes=block_sizes, steps_num=steps_num)

        # Compute loss
        agent_loss = self.compute_loss()

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        agent_loss.backward()

        # Clip gradients + optimizer step
        grad_norm = self.optimize_step()

        # Populate output queue
        self.episode_counter += 1

        # TODO: Log everything
        # self.log(agent_loss)
        # self.log(grad_norm)

    def get_work(self):
        pass

    def train(self):
        while True:
            # Wait until there's something to work on
            work = self.get_work()
            self.train_step()
