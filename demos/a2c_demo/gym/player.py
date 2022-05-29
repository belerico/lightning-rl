from typing import Optional, Tuple

import gym
import lightning as L
import numpy as np
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload

from demos.a2c_demo.agent.actor_critic import A2CAgent
from demos.a2c_demo.buffer.rollout import RolloutBuffer
from demos.a2c_demo.model.mlp import PolicyMLP


# Simple LightningWorker
class Player(L.LightningWork):
    """Wrapper around a gym environment to play the game.

    Args:
        environment (gym.Env): Gym environment
        agent (A2CAgent): Agent to interact with the environment
        output_path (str, optional): Output path. Defaults to "./".
    Raises:
        NotImplementedError: If the game mode is not supported
    """

    def __init__(
        self,
        environment_id: str,
        model_state_dict_path: Optional[Path] = None,
        agent_id: int = 0,
        **worker_kwargs
    ) -> None:
        super(Player, self).__init__(worker_kwargs)

        # Game
        self.replay_buffer = None  # Payload
        self.environment_id = environment_id
        self.observation_size, self.action_dim = Player.get_env_info(environment_id)

        # Agent
        self.agent_id = agent_id
        self.model_state_dict_path = model_state_dict_path

        # Misc
        self.step_counter = 0
        self.episode_counter = 0

    @torch.no_grad()
    def train_episode(self, agent: A2CAgent) -> RolloutBuffer:
        """Samples an episode for a single agent in training mode
        Returns:
            Tuple[np.ndarray, list]: Episode data and data sizes for each block of the buffer

        """
        environment = gym.make(self.environment_id)
        observation = environment.reset()
        game_done = False
        step_counter = 0
        buffer_data = {key: [] for key in RolloutBuffer.field_names()}
        while not game_done:
            observation_tensor = torch.from_numpy(observation).unsqueeze(0).float()
            actions, log_probs, values = agent.select_action(observation_tensor)
            log_probs = log_probs.numpy()
            values = values[0].numpy()
            actions = actions.numpy()
            env_actions = actions.flatten().tolist()
            env_actions = env_actions[0] if len(env_actions) == 1 else env_actions
            next_observation, reward, game_done, info = environment.step(env_actions)

            buffer_data["observations"].append(observation)
            buffer_data["values"].append(values)
            buffer_data["log_probs"].append(log_probs)
            buffer_data["actions"].append(actions)
            buffer_data["rewards"].append(reward)
            buffer_data["dones"].append([int(game_done)])

            observation = next_observation
            step_counter += 1

        for k, v in buffer_data.items():
            buffer_data[k] = np.array(v)

        buffer = RolloutBuffer.from_dict(buffer_data)
        return buffer

    @staticmethod
    def get_env_info(environment_id: str) -> Tuple[int, int]:
        environment = gym.make(environment_id)
        observation_size = environment.observation_space.shape[0]
        if isinstance(environment.action_space, gym.spaces.Discrete):
            action_dim = environment.action_space.n
        elif isinstance(environment.action_space, gym.spaces.MultiDiscrete):
            raise ValueError("MultiDiscrete spaces are not supported")
        elif isinstance(environment.action_space, gym.spaces.Box):
            action_dim = environment.action_space.shape[0]
        return observation_size, action_dim

    def run(self, signal: int):
        print("Player: playing episode {}".format(self.episode_counter))
        model = PolicyMLP(self.observation_size, [64, 64], self.action_dim)
        if self.model_state_dict_path.exists():
            print("Player: loading model from {}".format(self.model_state_dict_path))
            # self.model_state_dict_path.get(overwrite=True)
            model.load_state_dict(torch.load(self.model_state_dict_path))
        agent = A2CAgent(model=model, optimizer=None)

        # Play the game
        replay_buffer = self.train_episode(agent)
        print("Player: episode length: {}".format(len(replay_buffer)))
        self.replay_buffer = Payload(replay_buffer)

        self.episode_counter += 1
