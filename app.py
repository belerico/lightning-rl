import hydra
import lightning as L
import omegaconf
from hydra.experimental import compose, initialize
from lightning.runners import MultiProcessRuntime
from pympler import asizeof

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
        show_rl_info: bool = True,
    ):
        super().__init__()
        self.num_players = num_players
        self.max_episodes = max_episodes
        self.test_every_n_episodes = test_every_n_episodes
        self.show_rl_info = show_rl_info
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
        self.logger = TensorboardWork("./logs", parallel=True, run_once=False)
        self.gif_renderer = None
        if tester_cfg.save_rendering:
            self.gif_renderer = LitStreamlit(rendering_path=self.tester.rendering_path)

    def run(self):
        if not self.trainer.has_started or self.trainer.has_succeeded:
            self.players.run(self.trainer.episode_counter)
        if all(player.has_succeeded for player in self.players.players):
            self.trainer.run(self.players[0].episode_counter, self.players.buffers())
            if self.trainer.has_succeeded:
                self.trainer.metrics.update({"State/Size": asizeof.asizeof(self.state)})
                self.trainer.metrics.update({"Game/Episodes": self.trainer.episode_counter})
                self.logger.run(self.trainer.episode_counter, self.trainer.metrics)
        if self.trainer.episode_counter > 0 and self.trainer.episode_counter % self.test_every_n_episodes == 0:
            self.tester.run(self.trainer.episode_counter, test=True)
            if self.tester.has_succeeded:
                self.tester.test_metrics.update({"Game/Test episodes": self.tester.episode_counter})
                self.logger.run(self.tester.episode_counter, self.tester.test_metrics)
        if self.trainer.episode_counter >= self.max_episodes:
            self.logger.stop()
            self.tester.stop()
            self.trainer.stop()
            self.players.stop()

    def configure_layout(self):
        tabs = [{"name": "TB logs", "content": self.logger.url}]
        if self.gif_renderer is not None:
            tabs += [{"name": "Test GIF", "content": self.gif_renderer}]
        if self.show_rl_info:
            tabs += [
                {"name": "RL: intro", "content": "https://lilianweng.github.io/posts/2018-02-19-rl-overview/"},
                {
                    "name": "RL: policy gradients",
                    "content": "https://lilianweng.github.io/posts/2018-04-08-policy-gradient/",
                },
            ]
        return tabs


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
                show_rl_info=config.show_rl_info,
            )
        )
