from abc import ABC
from ctypes import Structure
from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    # PPO and A2C like algorithms
    # the data order follows the same order below
    observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        self.total_inputs_num = self.observations.shape[1]

    def __len__(self) -> int:
        """Returns the number of steeps in the ReplayBuffer"""
        return len(self.observations)

    def get_field(self, field_name: str) -> Any:
        return getattr(self, field_name)

    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in dataclasses.fields(cls) if f.type == np.ndarray]

    def to_array(self, dtype: np.dtype = np.dtype(np.float32)) -> Tuple[np.ndarray, List[int]]:
        """Converts the ReplayBuffer to a numpy flattened array"""
        data_sizes = []
        data = []
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            data.append(field.flatten())
            data_sizes.append(field.size)
        data = np.concatenate(data)
        assert data.size == sum(data_sizes)

        return data.astype(dtype), data_sizes

    @staticmethod
    def from_array(data: np.ndarray, steps_num: int, block_sizes: np.ndarray) -> "RolloutBuffer":
        """Creates a ReplayBuffer from a numpy array

        Args:
            data (np.ndarray): The data array
            steps_num (int): The number of steps in the current episode
            block_sizes (np.ndarray): The block sizes for observations, values, actions, log_probs and rewards.
                The sizes are cumulated.
        Returns:
            RolloutBuffer: a ReplayBuffer object with correctly reshaped obeservations, rewards, actions
            and actions probabilities
        """

        assert len(block_sizes) == len(RolloutBuffer.field_names())
        buffer_dict = {}
        block_counter = 0
        for idx, field_name in enumerate(RolloutBuffer.field_names()):
            values = data[block_counter : block_counter + block_sizes[idx]]
            buffer_dict[field_name] = values.reshape(steps_num, -1)
            block_counter += block_sizes[idx]
        return RolloutBuffer(**buffer_dict)

    def __getitem__(self, item: Union[int, List[int]]) -> Dict[str, Optional[torch.Tensor]]:
        if not isinstance(item, List):
            item = np.array([item])
        else:
            item = np.array(item)

        data = {}
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            data[field_name] = torch.from_numpy(field[item])
        return data

    def append(self, another_buffer: "RolloutBuffer") -> None:
        """Appends another ReplayBuffer to this one"""
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            another_field = another_buffer.get_field(field_name)
            field = np.concatenate((field, another_field))
            setattr(self, field_name, field)

    def shrink(self, total_size: int) -> None:
        """Shrinks the ReplayBuffer to the given size"""
        if self.__len__() > total_size:
            for field_name in RolloutBuffer.field_names():
                field = self.get_field(field_name)
                field = field[:total_size, :]
                setattr(self, field_name, field)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RolloutBuffer":
        """Creates a ReplayBuffer from a dictionary"""
        buffer_dict = {**data}
        return RolloutBuffer(
            **buffer_dict,
        )

    def add(self, data: Dict[str, np.ndarray]) -> None:
        """Adds data to the ReplayBuffer"""
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            if not isinstance(data[field_name], np.ndarray) and isinstance(data[field_name], torch.Tensor):
                data[field_name] = data[field_name].numpy()
            else:
                raise ValueError(f"{field_name} is neither a numpy array nor a torch tensor")
            setattr(self, field_name, np.concatenate((field, data[field_name])))
