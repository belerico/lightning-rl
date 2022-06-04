import base64
import os

import hydra
import lightning as L
import omegaconf
import streamlit as st
from hydra.experimental import compose, initialize
from lightning.frontend.stream_lit import StreamlitFrontend
from lightning.storage.path import Path

from demos.a2c_demo.logger.tensorboard import TensorboardWork
from demos.a2c_demo.player.player import Player, PlayersFlow
from demos.a2c_demo.trainer.trainer import Trainer


def render_gif(state) -> None:
    if os.path.exists(state.rendering_path) and os.listdir(state.rendering_path):
        gif = sorted(os.listdir(state.rendering_path), key=lambda x: x.split("_")[1], reverse=True)[0]
        file_ = open(os.path.join(state.rendering_path, gif), "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" >',
            unsafe_allow_html=True,
        )


class LitStreamlit(L.LightningFlow):
    def __init__(self, rendering_path: Path):
        super().__init__()
        self.rendering_path = rendering_path

    def run(self) -> None:
        if self.rendering_path.exists():
            self.rendering_path.get(overwrite=True)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_gif)


class A2CDemoFlow(L.LightningFlow):
    def __init__(
        self,
        player_cfg: omegaconf.DictConfig,
        tester_cfg: omegaconf.DictConfig,
        trainer_cfg: omegaconf.DictConfig,
        num_players: int = 2,
        max_episodes: int = 1000,
        save_rendering: bool = False,
        rendering_frequency: int = 10,
    ):
        super().__init__()
        self.num_players = num_players
        self.max_episodes = max_episodes
        self.save_rendering = save_rendering
        self.rendering_frequency = rendering_frequency
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
        if self.save_rendering:
            self.gif_renderer = LitStreamlit(rendering_path=self.tester.rendering_path)

    def run(self):
        self.players.run(self.trainer.episode_counter)
        for i in range(self.num_players):
            self.trainer.run(self.players.get_player(i).episode_counter, i, self.players.get_player(i).get_buffer())
        self.logger.run(self.trainer.episode_counter, self.trainer.metrics)
        if self.trainer.episode_counter > 0 and self.trainer.episode_counter % self.rendering_frequency == 0:
            self.tester.run(self.trainer.episode_counter, test=True)
        if self.trainer.episode_counter >= self.max_episodes:
            self.logger.stop()
            self.trainer.stop()
            self.players.stop()

    def configure_layout(self):
        tab_1 = {"name": "TB logs", "content": self.logger.url}
        if self.save_rendering:
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
                save_rendering=config.save_rendering,
                rendering_frequency=config.rendering_frequency,
            )
        )
