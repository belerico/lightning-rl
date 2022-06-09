import base64
import os
from typing import Optional

import lightning as L
from lightning.frontend.stream_lit import StreamlitFrontend
from lightning.storage.path import Path
from lightning.utilities.state import AppState

from lightning_rl import ROOT_DIR


def render_gif(state: AppState) -> None:
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(5000)
    _left, mid, _right = st.columns([0.2, 5, 0.2])
    with mid:
        if state is not None and os.path.exists(state.rendering_path):
            gifs = sorted(os.listdir(state.rendering_path), key=lambda x: x.split("_")[1], reverse=True)
            if len(gifs) > 0 and os.path.exists(os.path.join(state.rendering_path, gifs[0])):
                mid.image(os.path.join(state.rendering_path, gifs[0]), width=600, use_column_width=True)
            else:
                st.image(os.path.join(ROOT_DIR, "..", "images", "lightning.png"), width=400)
        else:
            st.image(os.path.join(ROOT_DIR, "..", "images", "lightning.png"), width=400)


class GIFRender(L.LightningFlow):
    """Simple StreamLit frontend. It shows the GIF generated by the tester.

    Args:
        rendering_path (Path): Path to the directory where the GIFs are stored.
    """

    def __init__(self, rendering_path: Path):
        super().__init__()
        self.rendering_path = rendering_path

    def run(self) -> None:
        if self.rendering_path.exists():
            self.rendering_path.get(overwrite=True)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_gif)
