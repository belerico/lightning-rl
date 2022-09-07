import os

import hydra
import lightning as L
import omegaconf
from hydra import compose, initialize_config_dir
from lightning_app.storage import Drive

from lightning_rl.frontend.config import EditConfUI
from lightning_rl.frontend.gif import GIFRender
from lightning_rl.logger.tensorboard import TensorboardWork
from lightning_rl.player.player import Player, PlayersFlow
from lightning_rl.trainer.trainer import Trainer
from lightning_rl.utils.utils import get_logger

logger = get_logger(__name__)


class RLTrainFlow(L.LightningFlow):
    def __init__(
        self,
        lightning_rl_drive: Drive,
        player_cfg: omegaconf.DictConfig,
        tester_cfg: omegaconf.DictConfig,
        trainer_cfg: omegaconf.DictConfig,
        num_players: int = 2,
        max_episodes: int = 1000,
        test_every_n_episodes: int = 10,
        show_rl_info: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lightning_rl_drive = lightning_rl_drive
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
            cache_calls=True,
            parallel=True,
        )
        self.tester: Player = hydra.utils.instantiate(tester_cfg, agent_id=0, cache_calls=True, parallel=True)
        self.players = PlayersFlow(self.num_players, player_cfg)
        self.logger = TensorboardWork(parallel=False, cache_calls=False)
        self.tester.log_dir = self.logger.tensorboard_log_dir
        self.train_ended = False

    def run(self):
        if not self.trainer.first_time_model_save:
            self.trainer.run(0)
        elif not self.trainer.has_started or self.trainer.has_succeeded:
            self.players.run(self.trainer.episode_counter, self.trainer.checkpoint_path)
        if all(player.has_succeeded for player in self.players.players):
            self.trainer.run(self.players[0].episode_counter, self.players.buffers())
            if self.trainer.metrics is not None:
                self.trainer.metrics.update({"Game/Train episodes": self.trainer.episode_counter})
            self.logger.run(
                self.trainer.episode_counter,
                self.trainer.metrics,
                self.lightning_rl_drive,
                self.trainer.checkpoint_path,
            )
        if self.trainer.episode_counter > 0 and self.trainer.episode_counter % self.test_every_n_episodes == 0:
            self.tester.run(
                self.trainer.episode_counter, self.trainer.checkpoint_path, self.lightning_rl_drive, test=True
            )
            if self.tester.has_succeeded:
                self.tester.test_metrics.update({"Game/Test episodes": self.tester.episode_counter})
                self.logger.run(self.tester.episode_counter, self.tester.test_metrics, self.lightning_rl_drive)
        if self.trainer.episode_counter >= self.max_episodes:
            self.logger.stop()
            self.tester.stop()
            self.trainer.stop()
            self.players.stop()
            self.train_ended = True


class RLDemoFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lightning_rl_drive = Drive("lit://lightning-rl-drive", allow_duplicates=True)
        self.edit_conf = EditConfUI(self.lightning_rl_drive)
        self.gif_renderer = GIFRender(self.lightning_rl_drive)
        self.train_flow = None
        self.train_flow_initialized = False

    def run(self):
        if self.edit_conf.train:
            if not self.train_flow_initialized:
                logger.info("Initializing hydra")
                with initialize_config_dir(os.path.abspath(os.path.join(self.edit_conf.tmp_hydra_dir, ".hydra"))):
                    config = compose(
                        "config.yaml", overrides=[k + "=" + v for k, v in self.edit_conf.hydra_overrides.items()]
                    )
                logger.info("Hydra configs composition finished")
                self.train_flow = RLTrainFlow(
                    self.lightning_rl_drive,
                    config.player,
                    config.tester,
                    config.trainer,
                    num_players=config.num_players,
                    max_episodes=config.max_episodes,
                    test_every_n_episodes=config.test_every_n_episodes,
                    show_rl_info=config.show_rl_info,
                )
                if self.gif_renderer.rendering_path is None:
                    rendering_path = os.path.join(
                        self.train_flow.logger.tensorboard_log_dir, self.train_flow.tester.local_rendering_path
                    )
                    self.gif_renderer.rendering_path = os.path.normpath(rendering_path)
                self.edit_conf.max_episodes = config.max_episodes
                self.train_flow_initialized = True
            else:
                self.train_flow.run()
                self.edit_conf.run(self.train_flow.trainer.episode_counter)
                self.edit_conf.train_ended = self.train_flow.train_ended
        if self.train_flow_initialized and self.train_flow.train_ended:
            self.edit_conf.train_ended = True

    def configure_layout(self):
        tabs = [{"name": "Configure your training", "content": self.edit_conf}]
        if self.train_flow_initialized:
            tabs += [{"name": "Training logs", "content": self.train_flow.logger.tensorboard_url}]
            if self.train_flow.tester.is_display_available:
                tabs += [{"name": "Learned agent", "content": self.gif_renderer}]
            if self.train_flow.show_rl_info:
                tabs += [
                    {"name": "RL: intro", "content": "https://lilianweng.github.io/posts/2018-02-19-rl-overview/"},
                    {
                        "name": "RL: policy gradients",
                        "content": "https://lilianweng.github.io/posts/2018-04-08-policy-gradient/",
                    },
                ]
        return tabs


if __name__ == "__main__":
    app = L.LightningApp(RLDemoFlow())
