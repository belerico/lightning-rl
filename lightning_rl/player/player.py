import os
import shutil
from typing import List, Optional, Tuple

import gym
import hydra
import lightning as L
import numpy as np
import omegaconf
import torch
from lightning.app.storage import Drive, Path, Payload
from lightning.app.structures import List as LightningList
from lightning_rl.agent.base import Agent

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
        gamma (np.ndarray): the discount factor. Default: 0.99.
        agent_id (int, optional): the agent id. Defaults to 0.
        log_dir (str, optional): the log directory of the logger. If specified then the renderings will be saved inside that folder.
            Defaults to None.
        local_rendering_path (str, optional): the local rendering path. Defaults to "./rendering.
        keep_last_n (int, optional): number of last gifs to keep. Defaults to -1.
    """

    def __init__(
        self,
        environment_id: str,
        agent_cfg: omegaconf.DictConfig,
        model_cfg: omegaconf.DictConfig,
        gamma: List[float] = [0.99],
        agent_id: int = 0,
        log_dir: Optional[str] = None,
        save_rendering: bool = False,
        local_rendering_path: str = "./rendering",
        keep_last_n: int = -1,
        **work_kwargs
    ) -> None:
        super(Player, self).__init__(**work_kwargs)

        # Private attributes
        self._environment_id = environment_id
        self._environment = gym.make(self._environment_id)
        self._environment.metadata["render_modes"] = ["rgb_array"]
        self._environment.metadata["render_fps"] = 120
        input_dim, action_dim = Player.get_env_info(environment_id)
        self._agent: Agent = hydra.utils.instantiate(
            agent_cfg,
            input_dim=input_dim,
            action_dim=action_dim,
            model_cfg=model_cfg,
            optimizer_cfg=None,
            distributed=False,
            _recursive_=False
        )
        if isinstance(gamma, list):
            gamma = np.array(gamma)
            if gamma.ndim <= 1:
                gamma = gamma.reshape(1, -1)
            else:
                raise ValueError("gamma must be a 1D array")
        self._gamma = gamma
        self._keep_last_n = keep_last_n

        # Public attributes
        setattr(self, "buffer_{}".format(agent_id), None)
        self.agent_id = agent_id
        self.episode_counter = 0
        self.log_dir = log_dir
        self.save_rendering = save_rendering
        self.local_rendering_path = local_rendering_path
        self.is_display_available = os.environ.get("DISPLAY", None) is not None
        os.makedirs(local_rendering_path, exist_ok=True)
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
    def test_episode(self, episode_counter, drive: Optional[Drive] = None) -> None:
        """Samples an episode for a single agent in training mode
        Returns:
            RolloutBuffer: Episode data in a RolloutBuffer
        """
        observation = self._environment.reset()
        game_done = False
        step_counter = 0
        frames = []
        total_reward = 0
        if self.save_rendering and not self.is_display_available:
            logger.warn("DISPLAY is not available, skipping rendering!")
        while not game_done:
            observation_tensor = torch.from_numpy(observation).unsqueeze(0).float()
            env_actions = self._agent.select_greedy_action(observation_tensor)
            env_actions = env_actions.numpy().flatten().tolist()
            env_actions = env_actions[0] if len(env_actions) == 1 else env_actions
            next_observation, reward, game_done, info = self._environment.step(env_actions)
            total_reward += reward
            observation = next_observation
            if self.save_rendering and self.is_display_available:
                frame = self._environment.render(mode="rgb_array")
                frames.append(frame)
            step_counter += 1
            self._environment.close()
        self.test_metrics["Test/sum_rew"] = total_reward
        if self.save_rendering and self.is_display_available:
            save_episode_as_gif(
                frames,
                path=self.local_rendering_path,
                episode_counter=episode_counter,
                keep_last_n=self._keep_last_n,
            )
            if drive is not None:
                if self.log_dir is None:
                    drive_rendering_path = self.local_rendering_path
                else:
                    drive_rendering_path = os.path.join(self.log_dir, self.local_rendering_path)
                drive_path = os.path.normpath(drive_rendering_path)
                os.makedirs(drive_path, exist_ok=True)
                shutil.copytree(self.local_rendering_path, drive_path, dirs_exist_ok=True)
                drive.put(drive_rendering_path)

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

    def run(self, signal: int, checkpoint_path: Path, drive: Optional[Drive] = None, test: Optional[bool] = False):
        if checkpoint_path.exists_remote():
            checkpoint_path.get(overwrite=True)
            self._agent.model.load_state_dict(torch.load(checkpoint_path))

        if test:
            logger.info("Tester-{}: testing episode {}".format(self.agent_id, self.episode_counter))
            self.test_episode(signal, drive)
            logger.info("Tester-{}: Sum of rewards: {:.4f}".format(self.agent_id, self.test_metrics["Test/sum_rew"]))
        else:
            logger.info("Player-{}: playing episode {}".format(self.agent_id, self.episode_counter))
            buffer = self.train_episode()
            logger.info("Player-{}: episode length: {}".format(self.agent_id, len(buffer)))
            setattr(self, "buffer_{}".format(self.agent_id), Payload(buffer))
        self.episode_counter += 1


class PlayersFlow(L.LightningFlow):
    def __init__(self, n_players: int, player_cfg: omegaconf.DictConfig):
        super().__init__()
        self.n_players = n_players
        self.episode_counter = 0
        self.players: LightningList[Player] = LightningList(
            *[
                hydra.utils.instantiate(
                    player_cfg,
                    agent_id=i,
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

    def run(self, signal: int, checkpoint_path: Path) -> None:
        for player in self.players:
            player.run(signal, checkpoint_path)

    def stop(self):
        for player in self.players:
            player.stop()
