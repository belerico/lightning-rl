import dataclasses
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class BaseBuffer(ABC):
    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in dataclasses.fields(cls) if f.type == np.ndarray]

    def get_field(self, field_name: str) -> Any:
        return getattr(self, field_name)

    def __len__(self) -> int:
        """Get the length of the buffer"""
        raise NotImplementedError

    def __getitem__(self, *args, **kwargs) -> Any:
        """Get the item at the index"""
        raise NotImplementedError

    def to_array(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def from_array(*args, **kwargs) -> "BaseBuffer":
        raise NotImplementedError

    @staticmethod
    def from_dict(*args, **kwargs) -> "BaseBuffer":
        raise NotImplementedError

    def add(self, data: Dict[str, np.ndarray]) -> None:
        """Add a new item to the buffer"""
        raise NotImplementedError

    def append(self, another_buffer: "BaseBuffer") -> None:
        """Appends another Buffer to this one"""
        raise NotImplementedError

    def shrink(self, total_size: int) -> None:
        """Shrinks the Buffer to the given size"""
        raise NotImplementedError


@dataclass
class RolloutBuffer(BaseBuffer):
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
        for field_name in RolloutBuffer.field_names():
            values = getattr(self, field_name)
            if len(values.shape) == 1:
                if field_name in ["rewards", "actions", "log_probs"]:
                    setattr(self, field_name, values.reshape(-1, self.__len__()).T)
                else:
                    setattr(self, field_name, values.reshape(self.__len__(), -1))

    def __len__(self) -> int:
        """Returns the number of steeps in the Buffer"""
        return len(self.observations)

    def to_array(self, dtype: np.dtype = np.dtype(np.float32)) -> Tuple[np.ndarray, List[int]]:
        """Converts the Buffer to a numpy flattened array"""
        data_sizes = []
        data = []
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            if field_name in ["rewards", "actions", "log_probs"]:
                data.append(field.T.flatten())
            else:
                data.append(field.flatten())
            data_sizes.append(field.size)
        data = np.concatenate(data)
        assert data.size == sum(data_sizes)

        return data.astype(dtype), data_sizes

    @staticmethod
    def from_array(data: np.ndarray, block_sizes: np.ndarray, steps_num: int) -> "RolloutBuffer":
        """Creates a Buffer from a numpy array

        Args:
            data (np.ndarray): The data array
            block_sizes (np.ndarray): The block sizes for observations, values, actions, log_probs and rewards.
                The sizes are cumulated.
        Returns:
            RolloutBuffer: a Buffer object with correctly reshaped obeservations, rewards, actions
            and actions probabilities
        """

        assert len(block_sizes) == len(RolloutBuffer.field_names())
        buffer_dict = {}
        block_counter = 0
        for idx, field_name in enumerate(RolloutBuffer.field_names()):
            values = data[block_counter : block_counter + block_sizes[idx]]
            if field_name in ["rewards", "actions", "log_probs"]:
                buffer_dict[field_name] = values.reshape(-1, steps_num).T
                print(values.reshape(-1, steps_num).T.shape)
            else:
                buffer_dict[field_name] = values.reshape(steps_num, -1)
            block_counter += block_sizes[idx]
        return RolloutBuffer(**buffer_dict)

    def __str__(self) -> str:
        """Returns a string representation of the Buffer"""
        representation = ["RolloutBuffer\n"]
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            representation.append(f"\t{field_name}: {field.shape}\n")
            print(f"{field_name}: {field.shape}")
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

    def add(self, data: Dict[str, np.ndarray]) -> None:
        """Adds data to the Buffer"""
        for field_name in RolloutBuffer.field_names():
            field = self.get_field(field_name)
            if not isinstance(data[field_name], np.ndarray) and isinstance(data[field_name], torch.Tensor):
                data[field_name] = data[field_name].numpy()
            else:
                raise ValueError(f"{field_name} is neither a numpy array nor a torch tensor")

            setattr(self, field_name, np.concatenate((field, data[field_name])))
