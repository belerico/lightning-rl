import os
from typing import Dict, List

import lightning as L
import streamlit as st
from lightning.frontend.stream_lit import StreamlitFrontend
from lightning.utilities.state import AppState

from lightning_rl import ROOT_DIR


def get_config_files() -> Dict[str, List[str]]:
    config_files = {}
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT_DIR, "configs")):
        for filename in filenames:
            if filename.endswith(".yaml"):
                dirname = os.path.basename(dirpath)
                if dirname not in config_files:
                    config_files[dirname] = []
                config_files[dirname].append(os.path.join(dirpath, filename))
    return config_files


def render(state: AppState):
    config_files = get_config_files()
    config_files = {k: config_files[k] for k in sorted(config_files)}
    st.title("Lightning RL Demo")
    st.write(
        "Edit your hydra configurations before training the agent. For more information see %s"
        % "https://hydra.cc/docs/intro/"
    )
    select_boxes = []
    selected_files = []
    texts = []
    for i, (dirname, files) in enumerate(config_files.items()):
        display_names = [os.path.basename(f) for f in sorted(files)]
        with st.expander(dirname.title(), expanded=i == 0):
            select_boxes.append(st.selectbox("Choose one config from:", display_names, key=i))
            file_to_read = None
            for f in files:
                if os.path.basename(f) == select_boxes[i]:
                    file_to_read = f
                    selected_files.append(file_to_read)
                    break
            num_lines = 0
            file_content = ""
            for line in open(file_to_read):
                num_lines += 1
                file_content += line
            texts.append(
                st.text_area(
                    "Edit " + dirname + "/" + select_boxes[i] + " to fit your needs",
                    file_content,
                    height=min(num_lines * 31, 500),
                    key=i,
                )
            )
    train = st.button("Train the agent")
    if train:
        for i, f in enumerate(selected_files):
            with open(f, "w") as f:
                f.write(texts[i])
        state.train = train


class EditConfUI(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train = False

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render)
