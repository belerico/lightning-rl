import hydra
import lightning as L
import omegaconf
from hydra.experimental import compose, initialize

from demos.a2c_demo.frontend.frontend import LitStreamlit
from demos.a2c_demo.logger.tensorboard import TensorboardWork
from demos.a2c_demo.player.player import Player, PlayersFlow
from demos.a2c_demo.trainer.trainer import Trainer


class A2CDemoFlow(L.LightningFlow):
    def __init__(
        self,
        player_cfg: omegaconf.DictConfig,
        tester_cfg: omegaconf.DictConfig,
        trainer_cfg: omegaconf.DictConfig,
        num_players: int = 2,
        max_episodes: int = 1000,
        test_every_n_episodes: int = 10,
    ):
        super().__init__()
        self.num_players = num_players
        self.max_episodes = max_episodes
        self.test_every_n_episodes = test_every_n_episodes
        input_dim, action_dim = Player.get_env_info(player_cfg.environment_id)
        self.trainer: Trainer = hydra.utils.instantiate(
            trainer_cfg,
            input_dim=input_dim,
            action_dim=action_dim,
            agent_id=0,
            num_players=num_players,
            run_once=True,
            parallel=True,
        )
        self.tester: Player = hydra.utils.instantiate(
            tester_cfg,
            model_state_dict_path=self.trainer.model_state_dict_path,
            agent_id=0,
            run_once=True,
            parallel=True,
        )
        self.players = PlayersFlow(
            self.num_players, player_cfg, model_state_dict_path=self.trainer.model_state_dict_path
        )
        self.logger = TensorboardWork("./logs", num_agents=1, parallel=True)
        self.gif_renderer = None
        if tester_cfg.save_rendering:
            self.gif_renderer = LitStreamlit(rendering_path=self.tester.rendering_path)

    def run(self):
        self.players.run(self.trainer.episode_counter)
        for i in range(self.num_players):
            self.trainer.run(self.players.get_player(i).episode_counter, i, self.players.get_player(i).get_buffer())
        self.logger.run(self.trainer.episode_counter, self.trainer.metrics)
        if self.trainer.episode_counter > 0 and self.trainer.episode_counter % self.test_every_n_episodes == 0:
            self.tester.run(self.trainer.episode_counter, test=True)
            self.logger.run(self.tester.episode_counter, self.tester.test_metrics)
        if self.trainer.episode_counter >= self.max_episodes:
            self.logger.stop()
            self.tester.stop()
            self.trainer.stop()
            self.players.stop()

    def configure_layout(self):
        tab_1 = {"name": "TB logs", "content": self.logger.url}
        if self.gif_renderer is not None:
            tab_2 = {"name": "Test GIF", "content": self.gif_renderer}
            return [tab_1, tab_2]
        else:
            return [tab_1]


if __name__ == "__main__":
    with initialize(config_path="./demos/a2c_demo/configs/"):
        config = compose(config_name="config.yaml")
        app = L.LightningApp(
            A2CDemoFlow(
                config.player,
                config.tester,
                config.trainer,
                num_players=config.num_players,
                max_episodes=config.max_episodes,
                test_every_n_episodes=config.test_every_n_episodes,
            )
        )
