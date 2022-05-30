import os
from typing import Optional, Tuple

import gym
import hydra
import lightning as L
import numpy as np
import omegaconf
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload
import torch.distributed as dist

from demos.a2c_demo.buffer.rollout import RolloutBuffer


# Simple LightningWorker
class Player(L.LightningWork):
    """Worker that wraps a gym environment and plays in it.

    Args:
        environment_id (str): the gym environment id
        agent_cfg (omegaconf.DictConfig): the agent configuration. The agent specifies the reinforcement learning
            algorithm to use. For this demo, we use the A2C algorithm (https://arxiv.org/abs/1602.01783).
        model_cfg (omegaconf.DictConfig): the model configuration. For this demo we have a simple linear model
            that outputs both the policy over actions and the value of the state.
        model_state_dict_path (Path): shared path to the model state dict.
        agent_id (int, optional): the agent id. Defaults to 0.
    """

    def __init__(
        self,
        environment_id: str,
        agent_cfg: omegaconf.DictConfig,
        model_cfg: omegaconf.DictConfig,
        model_state_dict_path: Path,
        agent_id: int = 0,
        **worker_kwargs
    ) -> None:
        super(Player, self).__init__(worker_kwargs)
        # Game
        self.replay_buffer = None  # Payload
        self.environment_id = environment_id
        self.input_dim, self.action_dim = Player.get_env_info(environment_id)

        model_cfg = model_cfg
        model = hydra.utils.instantiate(model_cfg, input_dim=self.input_dim, action_dim=self.action_dim)
        self._agent_cfg = agent_cfg
        self._agent = hydra.utils.instantiate(self._agent_cfg, model=model, optimizer=None)

        # Agent
        self.agent_id = agent_id
        self.model_state_dict_path = model_state_dict_path

        # Misc
        self.episode_counter = 0

    @torch.no_grad()
    def train_episode(self) -> RolloutBuffer:
        """Samples an episode for a single agent in training mode
        Returns:
            RolloutBuffer: Episode data in a RolloutBuffer

        """
        environment = gym.make(self.environment_id)
        observation = environment.reset()
        game_done = False
        step_counter = 0
        buffer_data = {key: [] for key in RolloutBuffer.field_names()}
        while not game_done:
            observation_tensor = torch.from_numpy(observation).unsqueeze(0).float()
            actions, log_probs, values = self._agent.select_action(observation_tensor)
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
        if self.model_state_dict_path.exists():
            print("Player: loading model from {}".format(self.model_state_dict_path))
            self.model_state_dict_path.get(overwrite=True)
            self._agent.model.load_state_dict(torch.load(self.model_state_dict_path))

        # Play the game
        replay_buffer = self.train_episode()
        print("Player: episode length: {}".format(len(replay_buffer)))
        self.replay_buffer = Payload(replay_buffer)
        self.episode_counter += 1
