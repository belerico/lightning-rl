import copy
from typing import List
import hydra
import lightning as L
from hydra.experimental import compose, initialize
import omegaconf

from demos.a2c_demo.player.player import Player
from demos.a2c_demo.trainer.trainer import Trainer


class TrainDeploy(L.LightningFlow):
    def __init__(
        self,
        player_cfg: omegaconf.DictConfig,
        trainer_cfg: omegaconf.DictConfig,
        num_agents: int = 3,
        max_episodes: int = 1000,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.max_episodes = max_episodes
        self._players = []
        self._trainers = []
        for i in range(num_agents):
            setattr(self, "player_{}".format(i), hydra.utils.instantiate(player_cfg, agent_id=i, run_once=True))
            setattr(self, "trainer_{}".format(i), hydra.utils.instantiate(trainer_cfg, agent_id=i, run_once=True))
            player = self.get_player(i)
            trainer = self.get_trainer(i)
            player.model_state_dict_path = trainer.model_state_dict_path
            trainer.action_dim = player.action_dim
            self._players.append(player)
            self._trainers.append(trainer)

    def get_all_players(self) -> List[Player]:
        return [self.get_player(i) for i in range(self.num_agents)]

    def get_player(self, agent_id: int) -> Player:
        return getattr(self, "player_{}".format(agent_id))

    def get_all_trainers(self) -> List[Trainer]:
        return [self.get_trainer(i) for i in range(self.num_agents)]

    def get_trainer(self, agent_id: int) -> Trainer:
        return getattr(self, "trainer_{}".format(agent_id))

    def run(self):
        if not any([self._trainers[i].has_started for i in range(self.num_agents)]) or all(
            [not self._trainers[i].is_running and self._trainers[i].has_succeeded for i in range(self.num_agents)]
        ):
            for i in range(self.num_agents):
                self._players[i].run(self._trainers[i].episode_counter)
        if all([not self._players[i].is_running and self._players[i].has_succeeded for i in range(self.num_agents)]):
            for i in range(self.num_agents):
                self._trainers[i].run(self._players[i].episode_counter, self._players[i].replay_buffer)
        if self._trainers[0].episode_counter >= self.max_episodes:
            for i in range(self.num_agents):
                self._trainers[i].stop()
                self._players[i].stop()


if __name__ == "__main__":
    with initialize(config_path="./demos/a2c_demo/configs/"):
        config = compose(config_name="config.yaml")
        app = L.LightningApp(TrainDeploy(config.player, config.trainer, max_episodes=500))
