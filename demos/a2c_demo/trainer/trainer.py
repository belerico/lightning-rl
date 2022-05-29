import os
from typing import Optional

import lightning as L
import torch
from lightning.storage.path import Path
from lightning.storage.payload import Payload

from demos.a2c_demo.agent.actor_critic import A2CAgent
from demos.a2c_demo.model.mlp import PolicyMLP


# Simple LightningWorker
class Trainer(L.LightningWork):
    """Gym environment worker

    Args:
        agent (A2CAgent): Agent to interact with the environment
        agent_id (int): Agent id.
        agent_data_path (Path): Path to shared agent data.
        data_sizes_path (Path): PAth to shared agent sizes.
    Raises:
        NotImplementedError: If the game mode is not supported
    """

    def __init__(
        self,
        action_dim: Optional[int] = None,
        agent_id: int = 0,
        agent_state_dir: str = "./agent_state",
        **worker_kwargs
    ) -> None:
        super(Trainer, self).__init__(worker_kwargs)

        # Agent
        self.agent_id = agent_id
        self.action_dim = action_dim

        # Misc
        self.episode_counter = 0

        # Path to save model and optim state
        self.agent_state_dir = agent_state_dir
        os.makedirs(agent_state_dir, exist_ok=True)
        self.model_state_dict_path = Path(os.path.join(agent_state_dir, "model_state_dict_" + str(agent_id)))
        self.optimizer_state_dict_path = os.path.join(agent_state_dir, "optimizer_state_dict_" + str(agent_id))

    def run(self, signal: int, buffer: Payload):
        print("Trainer: training episode {}".format(self.episode_counter))
        buffer = buffer.value

        model = PolicyMLP(buffer.observations.shape[1], [64, 64], self.action_dim)
        if self.model_state_dict_path.exists():
            print("Trainer: loading model state dict")
            model.load_state_dict(torch.load(self.model_state_dict_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if os.path.exists(self.optimizer_state_dict_path):
            print("Trainer: loading optimizer state dict")
            optimizer.load_state_dict(torch.load(self.optimizer_state_dict_path))
        agent = A2CAgent(model=model, optimizer=optimizer, batch_size=256, clip_gradients=200)

        agent.replay_buffer = buffer
        print("Trainer: training agent")
        agent_loss, sum_rewards = agent.train_step()
        print("Trainer: Loss: {:.4f}, Sum of rewards: {:.4f}".format(agent_loss, sum_rewards))

        # Save model and optim state
        print("Trainer: saving model state dict")
        torch.save(model.state_dict(), self.model_state_dict_path)
        torch.save(optimizer.state_dict(), self.optimizer_state_dict_path)

        self.episode_counter += 1
        return
