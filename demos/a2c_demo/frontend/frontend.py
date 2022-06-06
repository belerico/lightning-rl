import base64
import os

import lightning as L
import streamlit as st
from lightning.frontend.stream_lit import StreamlitFrontend
from lightning.storage.path import Path


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
