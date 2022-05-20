import os
from typing import Tuple

import gym
import lightning as L
import numpy as np
import torch
from lightning.storage.path import Path

from demos.a2c_demo.agent.actor_critic import A2CAgent
from demos.a2c_demo.buffer.rollout import RolloutBuffer
from demos.a2c_demo.model.mlp import PolicyMLP


# Simple LightningWorker
class GymWorker(L.LightningWork):
    """Gym environment worker

    Args:
        environment (gym.Env): Gym environment
        agent (A2CAgent): Agent to interact with the environment
        agent_id (int, optional): Agent id.
        output_path (str, optional): Output path. Defaults to "./".
    Raises:
        NotImplementedError: If the game mode is not supported
    """

    def __init__(
        self,
        environment_id: str,
        max_episodes: int = 2,
        agent_id: int = 0,
        output_dir="./",
        **worker_kwargs
    ) -> None:
        super(GymWorker, self).__init__(worker_kwargs)

        # Game
        self.environment_id = environment_id

        # Agent
        self.agent_id = agent_id

        # Misc
        self.max_episodes = max_episodes
        self.step_counter = 0
        self.episode_counter = 0
        self.output_dir = output_dir
        self.output_data_path = Path(os.path.join(output_dir, "agent_data_" + str(agent_id)))
        self.output_sizes_path = Path(os.path.join(output_dir, "agent_sizes_" + str(agent_id)))

    @torch.no_grad()
    def train_episode(self, agent: A2CAgent) -> Tuple[np.ndarray, np.ndarray]:
        """Samples an episode for a single agent in training mode
        Returns:
            Tuple[np.ndarray, list]: Episode data and data sizes for each block of the buffer

        """
        environment = gym.make(self.environment_id)

        observation_list = []
        values_list = []
        action_list = []
        log_prob_list = []
        reward_list = []
        dones_list = []
        observation = environment.reset()
        game_done = False
        self.step_counter = 0
        while not game_done:
            observation_tensor = torch.from_numpy(observation).unsqueeze(0).float()

            # actions: torch.Tensor = 1
            # log_probs: torch.Tensor = 1
            # values: torch.Tensor = N x 1
            actions, log_probs, values = agent.select_action(observation_tensor)

            log_prob_list.append(log_probs.numpy())
            action_list.append(actions.numpy())
            values_list.append(values[:, 0].numpy())
            observation_list.append(observation)

            # Perform a step in the environment
            next_observation, reward, game_done, info = environment.step(actions.numpy()[0])

            dones_list.append(game_done)
            reward_list.append([reward])

            observation = next_observation
            self.step_counter += 1

        self.episode_counter += 1
        replay_buffer = RolloutBuffer(
            observations=np.array(observation_list),
            values=np.array(values_list),
            actions=np.array(action_list),
            log_probs=np.array(log_prob_list),
            rewards=np.array(reward_list),
            dones=np.array(dones_list),
        )
        data, data_sizes = replay_buffer.to_array()
        return data, np.array(data_sizes).astype(np.int32)

    def run(self):
        while self.episode_counter < self.max_episodes or self.max_episodes < 0:
            environment = gym.make(self.environment_id)
            observation_size = environment.observation_space.shape[0]
            if isinstance(environment.action_space, gym.spaces.Discrete):
                action_dim = environment.action_space.n
            elif isinstance(environment.action_space, gym.spaces.MultiDiscrete):
                raise ValueError("MultiDiscrete spaces are not supported")
            elif isinstance(environment.action_space, gym.spaces.Box):
                action_dim = environment.action_space.shape[0]

            # TODO: load model state from checkpoint after the trainer has optimized the model
            model = PolicyMLP(observation_size, [64, 64], action_dim)
            agent = A2CAgent(model=model, optimizer=None)
            data, sizes = self.train_episode(agent)
            with open(self.output_data_path, "wb") as f:
                f.write(data.tobytes())
            with open(self.output_sizes_path, "wb") as f:
                f.write(sizes.tobytes())
