import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import lightning as L
import numpy as np
import torch
from lightning.storage.payload import Payload

from . import logger


class BufferWork(L.LightningWork):
    def __init__(self, buffers_to_receive: int, **worker_kwargs):
        super().__init__(**worker_kwargs)
        self.buffer = None
        self._buffer: RolloutBuffer = None
        self.buffers_to_receive = buffers_to_receive
        self._received_buffers = []
        self.episode_counter = 0

    def run(self, signal: int, agent_id: int, agent_buffer: Optional[Payload] = None):
        if agent_buffer is not None and agent_id not in self._received_buffers:
            logger.info("Received buffer from agent {}, length: {}".format(agent_id, len(agent_buffer.value)))
            self._received_buffers.append(agent_id)
            agent_buffer = agent_buffer.value
            if self._buffer is None:
                self._buffer = agent_buffer
            else:
                self._buffer.append(agent_buffer)
        if len(self._received_buffers) == self.buffers_to_receive:
            logger.info("Received all buffers. Total buffer length: {}".format(len(self._buffer)))
            self.buffer = Payload(self._buffer)
            self._buffer = None
            self._received_buffers = []
            self.episode_counter += 1


@dataclass
class RolloutBuffer:
    observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    returns: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None

    def __post_init__(self):
        self.total_inputs_num = self.observations.shape[1]
        for field_name in RolloutBuffer.field_names():
            values = getattr(self, field_name)
            if values is not None and len(values.shape) == 1:
                if field_name in ["rewards", "actions", "log_probs"]:
                    setattr(self, field_name, values.reshape(-1, self.__len__()).T)
                else:
                    setattr(self, field_name, values.reshape(self.__len__(), -1))

    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in dataclasses.fields(cls) if f.type == np.ndarray or f.type == Optional[np.ndarray]]

    def get_field(self, field_name: str) -> Any:
        return getattr(self, field_name)

    def __len__(self) -> int:
        """Returns the number of steeps in the Buffer"""
        return len(self.observations)

    def __str__(self) -> str:
        """Returns a string representation of the Buffer"""
        representation = ["RolloutBuffer\n"]
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            representation.append(f"\t{field_name}: {field.shape}\n")
        return "".join(representation)

    def __getitem__(self, item: Union[int, List[int]]) -> Dict[str, Optional[torch.Tensor]]:
        if not isinstance(item, List):
            item = np.array([item])
        else:
            item = np.array(item)

        data = {}
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            if field_name == "observations":
                observation = torch.from_numpy(field[item])
                data["observations"] = observation
                data["lstm_states"] = None
            else:
                data[field_name] = torch.from_numpy(field[item])
        return data

    def append(self, another_buffer: "RolloutBuffer") -> None:
        """Appends another Buffer to this one"""
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            another_field = another_buffer.get_field(field_name)
            field = np.concatenate((field, another_field))
            setattr(self, field_name, field)

    def shrink(self, total_size: int) -> None:
        """Shrinks the Buffer to the given size"""
        if self.__len__() > total_size:
            for field_name in RolloutBuffer.field_names():
                field = self.get_field(field_name)
                field = field[:total_size, :]
                setattr(self, field_name, field)

    @staticmethod
    def from_dict(
        data: Dict[str, Any],
    ) -> "RolloutBuffer":
        """Creates a Buffer from a dictionary"""
        return RolloutBuffer(**data)

    def compute_returns_and_advatages(
        self,
        gamma: np.ndarray = np.array([0.99]),
        normalize_returns: bool = True,
    ):
        """Compute discounted returns

        Args:
            rewards (np.ndarray): array of rewards of every time step. It has a shape of
                TxNr, where `T` is the number of time steps, while `Nr` is the number of
                rewards.

        Returns:
        np.ndarray: the array of discounted rewards of shape Tx1 if `reduction` is "sum" or "mean",
        of shape TxNr otherwise.
        """
        n_steps = self.rewards.shape[0]
        returns = np.zeros(self.rewards.shape)
        R = np.zeros(gamma.shape)

        for step in reversed(range(n_steps)):
            r = self.rewards[step, :]
            R = r + gamma * R * (1 - self.dones[step, :])
            returns[step, :] = R
        returns = np.sum(returns, axis=1, keepdims=True)
        if normalize_returns:
            returns = (returns - np.mean(returns, axis=0, keepdims=True)) / (
                np.std(returns, ddof=1, axis=0, keepdims=True) + 1e-8
            )

        self.returns = returns
        self.advantages = returns - self.values
