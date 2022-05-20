import os
from typing import Optional, Tuple

import gym
import lightning as L
import numpy as np
import torch
from lightning.storage.path import Path

from demos.a2c_demo.agent.actor_critic import A2CAgent
from demos.a2c_demo.buffer.rollout import RolloutBuffer


# Simple LightningWorker
class TrainWorker(L.LightningWork):
    """Gym environment worker

    Args:
        agent (A2CAgent): Agent to interact with the environment
        agent_id (int): Agent id.
        agent_data_path (Path): Path to shared agent data.
        agent_sizes_path (Path): PAth to shared agent sizes.
    Raises:
        NotImplementedError: If the game mode is not supported
    """

    def __init__(
        self,
        max_episodes: int = 2,
        agent_id: int = 0,
        agent_data_path: Optional[Path] = None,
        agent_sizes_path: Optional[Path] = None,
        **worker_kwargs
    ) -> None:
        super(TrainWorker, self).__init__(worker_kwargs)

        # Agent
        self.agent_id = agent_id

        # Misc
        self.max_episodes = max_episodes
        self.episode_counter = 0
        self.agent_data_path = agent_data_path
        self.agent_sizes_path = agent_sizes_path

    def run(self):
        while self.episode_counter < self.max_episodes:
            with open(self.agent_data_path, "rb") as f:
                agent_data = np.frombuffer(f.read(), dtype=np.float32)
                print("Agent data", agent_data.shape)
            with open(self.agent_sizes_path, "rb") as f:
                agent_sizes = np.frombuffer(f.read(), dtype=np.int32)
                print("Agent sizes", agent_sizes.shape)
            buffer = RolloutBuffer.from_array(agent_data, agent_sizes[-1], agent_sizes)
            print("Buffer", buffer[0])
            self.episode_counter += 1

