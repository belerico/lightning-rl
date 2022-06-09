from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyMLP(torch.nn.Module):
    """Generic one layer MLP model with ReLU activation in each hidden layer. It can adapt action dimensions.

    Args:
        input_dim (int): Input dimension of the model.
        hidden_dims (List[int]): dimension of the hidden layers.
        action_dims (int): action dimensions.
        act_fun: Optional[torch.nn.Module]: the activation function. Default is torch.nn.ReLU()
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int], action_dim: int, act_fun: Optional[torch.nn.Module] = None
    ):

        super(PolicyMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        layers_list = [nn.Linear(self.input_dim, hidden_dims[0]), act_fun or nn.ReLU()]
        for dim_i in range(len(hidden_dims) - 1):
            layers_list.append(nn.Linear(hidden_dims[dim_i], hidden_dims[dim_i + 1]))
            layers_list.append(act_fun or nn.ReLU())

        self.backbone = nn.Sequential(*layers_list)
        self.critic = nn.Linear(hidden_dims[-1], 1)
        self.actor = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            action_probs (torch.Tensor): Action probabilities.
            state_value (torch.Tensor): State value.

        """
        x = self.backbone(x)
        logits = self.actor(x)
        state_value = self.critic(x)
        return (F.softmax(logits, dim=-1), state_value)
