import hydra
import lightning as L
from hydra.experimental import compose, initialize

from demos.a2c_demo.player.player import Player
from demos.a2c_demo.trainer.trainer import Trainer


class TrainDeploy(L.LightningFlow):
    def __init__(self, player: Player, trainer: Trainer, max_episodes: int = 1000):
        super().__init__()
        player.model_state_dict_path = trainer.model_state_dict_path
        trainer.action_dim = player.action_dim
        self.player = player
        self.trainer = trainer
        self.max_episodes = max_episodes

    def run(self):
        if not self.trainer.has_started or (not self.trainer.is_running and self.trainer.has_succeeded):
            self.player.run(self.trainer.episode_counter)
        if not self.player.is_running and self.player.has_succeeded:
            self.trainer.run(self.player.episode_counter, self.player.replay_buffer)
        if self.trainer.episode_counter >= self.max_episodes:
            self.player.stop()
            self.trainer.stop()


if __name__ == "__main__":
    with initialize(config_path="./demos/a2c_demo/configs/"):
        config = compose(config_name="config.yaml")
        player = hydra.utils.instantiate(config.player, run_once=True)
        trainer = hydra.utils.instantiate(config.trainer, run_once=True)
        app = L.LightningApp(TrainDeploy(player, trainer, max_episodes=500))
