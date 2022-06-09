import os
from typing import List, Optional, Tuple, Union

import gym
import hydra
import lightning as L
import numpy as np
import omegaconf
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload
from lightning.structures import List as LightningList

from lightning_rl.buffer.rollout import RolloutBuffer
from lightning_rl.utils.viz import save_episode_as_gif

from . import logger


class Player(L.LightningWork):
    """Worker that wraps a gym environment and plays in it.

    Args:
        environment_id (str): the gym environment id
        agent_cfg (omegaconf.DictConfig): the agent configuration. The agent specifies the reinforcement learning
            algorithm to use. For this demo, we use the A2C algorithm (https://arxiv.org/abs/1602.01783).
        model_cfg (omegaconf.DictConfig): the model configuration. For this demo we have a simple linear model
            that outputs both the policy over actions and the value of the state.
        model_state_dict_path (Path): shared path to the model state dict.
        gamma (np.ndarray): the discount factor. Default: 0.99.
        agent_id (int, optional): the agent id. Defaults to 0.
        save_rendering (bool, optional): whether to save the rendering. Defaults to False.
        keep_last_n (int, optional): number of last gifs to keep. Defaults to -1.
        rendering_path (Union[Path, str], optional): path to the directory where to save the rendering. Defaults to None.
    """

    def __init__(
        self,
        environment_id: str,
        agent_cfg: omegaconf.DictConfig,
        model_cfg: omegaconf.DictConfig,
        model_state_dict_path: Path,
        gamma: List[float] = [0.99],
        agent_id: int = 0,
        save_rendering: bool = False,
        keep_last_n: int = -1,
        rendering_path: Optional[Union[Path, str]] = None,
        **work_kwargs
    ) -> None:
        super(Player, self).__init__(work_kwargs)
        setattr(self, "buffer_{}".format(agent_id), None)
        self.environment_id = environment_id
        self._environment = gym.make(self.environment_id)
        self._environment.metadata["render_modes"] = ["rgb_array"]
        self._environment.metadata["render_fps"] = 120
        input_dim, action_dim = Player.get_env_info(environment_id)
        model = hydra.utils.instantiate(model_cfg, input_dim=input_dim, action_dim=action_dim)
        self._agent = hydra.utils.instantiate(agent_cfg, model=model, optimizer=None)
        if isinstance(model_state_dict_path, str):
            model_state_dict_path = Path(model_state_dict_path)
        self.model_state_dict_path = model_state_dict_path
        if isinstance(gamma, list):
            gamma = np.array(gamma)
            if gamma.ndim <= 1:
                gamma = gamma.reshape(1, -1)
            else:
                raise ValueError("gamma must be a 1D array")
        self._gamma = gamma
        self.agent_id = agent_id
        self.episode_counter = 0
        self.save_rendering = save_rendering
        self._keep_last_n = keep_last_n
        if rendering_path is not None:
            if isinstance(rendering_path, str):
                os.makedirs(rendering_path, exist_ok=True)
                rendering_path = Path(rendering_path)
        self.rendering_path = rendering_path
        self.test_metrics = {}

    def get_buffer(self) -> Optional[Payload]:
        return getattr(self, "buffer_{}".format(self.agent_id))

    @torch.no_grad()
    def train_episode(self) -> RolloutBuffer:
        """Samples an episode for a single agent in training mode
        Returns:
            RolloutBuffer: Episode data in a RolloutBuffer

        """
        observation = self._environment.reset()
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
            next_observation, reward, game_done, info = self._environment.step(env_actions)

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
        buffer.compute_returns_and_advatages(gamma=self._gamma)
        return buffer

    @torch.no_grad()
    def test_episode(self, episode_counter) -> None:
        """Samples an episode for a single agent in training mode
        Returns:
            RolloutBuffer: Episode data in a RolloutBuffer

        """
        observation = self._environment.reset()
        game_done = False
        step_counter = 0
        frames = []
        total_reward = 0
        while not game_done:
            observation_tensor = torch.from_numpy(observation).unsqueeze(0).float()
            env_actions = self._agent.select_greedy_action(observation_tensor)
            env_actions = env_actions.numpy().flatten().tolist()
            env_actions = env_actions[0] if len(env_actions) == 1 else env_actions
            next_observation, reward, game_done, info = self._environment.step(env_actions)
            total_reward += reward
            observation = next_observation
            if self.save_rendering:
                frame = self._environment.render(mode="rgb_array")
                frames.append(frame)
            step_counter += 1
        self._environment.close()
        self.test_metrics["Test/sum_rew"] = total_reward
        if self.save_rendering:
            save_episode_as_gif(
                frames, path=self.rendering_path, episode_counter=episode_counter, keep_last_n=self._keep_last_n
            )

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

    def run(self, signal: int, test: Optional[bool] = False):
        if self.model_state_dict_path.exists():
            self.model_state_dict_path.get(overwrite=True)
            self._agent.model.load_state_dict(torch.load(self.model_state_dict_path))

        if test:
            logger.info("Tester-{}: testing episode {}".format(self.agent_id, self.episode_counter))
            self.test_episode(signal)
            logger.info("Tester-{}: Sum of rewards: {:.4f}".format(self.agent_id, self.test_metrics["Test/sum_rew"]))
        else:
            logger.info("Player-{}: playing episode {}".format(self.agent_id, self.episode_counter))
            buffer = self.train_episode()
            logger.info("Player-{}: episode length: {}".format(self.agent_id, len(buffer)))
            setattr(self, "buffer_{}".format(self.agent_id), Payload(buffer))
        self.episode_counter += 1


class PlayersFlow(L.LightningFlow):
    def __init__(self, n_players: int, player_cfg: omegaconf.DictConfig, model_state_dict_path: Path):
        super().__init__()
        self.n_players = n_players
        self.episode_counter = 0
        self.players: LightningList[Player] = LightningList(
            *[
                hydra.utils.instantiate(
                    player_cfg,
                    agent_id=i,
                    model_state_dict_path=model_state_dict_path,
                    run_once=True,
                    parallel=True,
                )
                for i in range(self.n_players)
            ]
        )

    def __getitem__(self, key: int) -> Player:
        return self.players[key]

    def buffers(self) -> List[Payload]:
        return [player.get_buffer() for player in self.players]

    def run(self, signal: int) -> None:
        for player in self.players:
            player.run(signal)

    def stop(self):
        for player in self.players:
            player.stop()
