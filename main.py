from typing import List

import hydra
import lightning as L
import omegaconf
from hydra.experimental import compose, initialize

from demos.a2c_demo.logger.tensorboard import TensorboardWork
from demos.a2c_demo.player.player import Player, PlayersFlow
from demos.a2c_demo.trainer.trainer import Trainer


class A2CDemoFlow(L.LightningFlow):
    def __init__(
        self,
        player_cfg: omegaconf.DictConfig,
        trainer_cfg: omegaconf.DictConfig,
        num_agents: int = 1,
        max_episodes: int = 1000,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.max_episodes = max_episodes
        input_dim, action_dim = Player.get_env_info(player_cfg.environment_id)
        self.trainer: Trainer = hydra.utils.instantiate(
            trainer_cfg,
            input_dim=input_dim,
            action_dim=action_dim,
            agent_id=0,
            run_once=True,
            parallel=True,
        )
        self.players = PlayersFlow(
            self.num_agents, player_cfg, model_state_dict_path=self.trainer.model_state_dict_path
        )
        self.logger = TensorboardWork("./logs", num_agents=self.num_agents, parallel=True)

    def run(self):
        self.players.run(self.trainer.episode_counter)
        self.trainer.run(self.players.buffer_work.episode_counter, self.players.buffer_work.buffer)
        self.logger.run(self.trainer.episode_counter, self.trainer.metrics)
        if self.trainer.episode_counter >= self.max_episodes:
            self.logger.stop()
            self.trainer.stop()
            self.players.stop()

    def configure_layout(self):
        tab_1 = {"name": "TB logs", "content": self.logger.url}
        return tab_1


if __name__ == "__main__":
    with initialize(config_path="./demos/a2c_demo/configs/"):
        config = compose(config_name="config.yaml")
        app = L.LightningApp(A2CDemoFlow(config.player, config.trainer, max_episodes=500))
