from typing import List

import hydra
import lightning as L
import omegaconf
from hydra.experimental import compose, initialize
from lightning.storage.path import Path
from demos.a2c_demo.logger.tensorboard import TensorboardWork

from demos.a2c_demo.optimizer.optimizer import Optimizer
from demos.a2c_demo.player.player import Player
from demos.a2c_demo.trainer.trainer import Trainer


class A2CDemoFlow(L.LightningFlow):
    def __init__(
        self,
        player_cfg: omegaconf.DictConfig,
        trainer_cfg: omegaconf.DictConfig,
        num_agents: int = 4,
        max_episodes: int = 1000,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.max_episodes = max_episodes
        self._players: List[Player] = []
        self._trainers: List[Trainer] = []
        input_dim, action_dim = Player.get_env_info(player_cfg.environment_id)
        self.optimizer = Optimizer(
            input_dim,
            action_dim,
            num_agents,
            trainer_cfg.model_cfg,
            trainer_cfg.optimizer_cfg,
            run_once=True,
            parallel=True,
        )
        for i in range(num_agents):
            setattr(
                self,
                "player_{}".format(i),
                hydra.utils.instantiate(
                    player_cfg,
                    agent_id=i,
                    model_state_dict_path=self.optimizer.model_state_dict_path,
                    run_once=True,
                    parallel=True,
                ),
            )
            setattr(
                self,
                "trainer_{}".format(i),
                hydra.utils.instantiate(
                    trainer_cfg,
                    input_dim=input_dim,
                    action_dim=action_dim,
                    agent_id=i,
                    model_state_dict_path=self.optimizer.model_state_dict_path,
                    run_once=True,
                    parallel=True,
                ),
            )
            player = self.get_player(i)
            trainer = self.get_trainer(i)
            self._players.append(player)
            self._trainers.append(trainer)
        self.logger = TensorboardWork("./logs", num_agents=self.num_agents, parallel=True)

    def get_all_players(self) -> List[Player]:
        return [self.get_player(i) for i in range(self.num_agents)]

    def get_player(self, agent_id: int) -> Player:
        return getattr(self, "player_{}".format(agent_id))

    def get_all_trainers(self) -> List[Trainer]:
        return [self.get_trainer(i) for i in range(self.num_agents)]

    def get_trainer(self, agent_id: int) -> Trainer:
        return getattr(self, "trainer_{}".format(agent_id))

    def run(self):
        if not any([self._trainers[i].has_started for i in range(self.num_agents)]) or self.optimizer.done:
            self.optimizer.done = False
            for i in range(self.num_agents):
                self._players[i].run(self._trainers[i].episode_counter)
        for i in range(self.num_agents):
            self._trainers[i].run(self._players[i].episode_counter, self._players[i].replay_buffer)
        for i in range(self.num_agents):
            self.optimizer.run(
                self._trainers[i].episode_counter, self._trainers[i].agent_id, self._trainers[i].gradients
            )
        for i in range(self.num_agents):
            self.logger.run(self._trainers[i].episode_counter, self._trainers[i].metrics)
        if self._trainers[0].episode_counter >= self.max_episodes:
            self.logger.stop()
            self.optimizer.stop()
            for i in range(self.num_agents):
                self._trainers[i].stop()
                self._players[i].stop()

    def configure_layout(self):
        tab_1 = {"name": "TB logs", "content": "http://localhost:6006/"}
        return tab_1


if __name__ == "__main__":
    with initialize(config_path="./demos/a2c_demo/configs/"):
        config = compose(config_name="config.yaml")
        app = L.LightningApp(A2CDemoFlow(config.player, config.trainer, max_episodes=500))
