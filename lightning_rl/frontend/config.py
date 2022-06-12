import base64
import os
import shutil
import tempfile
from typing import Dict, List, Optional
from urllib.parse import quote, urljoin

import lightning as L
import streamlit as st
from lightning.core.constants import APP_SERVER_HOST, APP_SERVER_PORT
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


def title(logo_path: str):
    st.markdown(
        """
        <style>
        .logo-text {
            font-weight: 700;
            font-size: 50px;
            display:inline-block;
            vertical-align:middle;
        }
        .logo-img {
            height: 50px;
            padding-right: 10px;
            width: auto;
            vertical-align:middle;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}">
            <div class="logo-text">Lightning RL Demo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(state: Optional[AppState] = None):
    st.set_page_config(layout="wide")
    if (state is not None and not state.train) or state is None:
        title(os.path.join(ROOT_DIR, "..", "images", "logo.png"))
        st.write(
            "Edit your hydra configurations before training the agent. For more information see %s"
            % "https://hydra.cc/docs/intro/"
        )
        config_files = get_config_files()
        config_files = {k: config_files[k] for k in sorted(config_files)}
        select_boxes = []
        selected_files = []
        texts = []
        hydra_overrides = {}
        _, mid_left, mid_right, _ = st.columns([0.05, 0.45, 0.45, 0.05])
        for i, (dirname, files) in enumerate(config_files.items()):
            if dirname == "configs":
                dirname = "Main config"
            display_names = [os.path.basename(f) for f in files]
            if "default.yaml" in display_names:
                display_names.insert(0, display_names.pop(display_names.index("default.yaml")))
            if i % 2 == 0:
                col_context = mid_left
            else:
                col_context = mid_right
            with col_context:
                with st.expander(dirname.title(), expanded=False):
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
                    if dirname != "Main config":
                        hydra_overrides[dirname] = select_boxes[i]
        with mid_left:
            train = st.button("Train the agent")
        if train and state is not None:
            os.makedirs(os.path.join(state.tmp_hydra_dir, ".hydra"), exist_ok=True)
            for i, (_, files) in enumerate(config_files.items()):
                hydra_dir = os.path.basename(os.path.dirname(files[0]))
                dirname = os.path.join(state.tmp_hydra_dir, ".hydra")
                if hydra_dir != "configs":
                    dirname = os.path.join(dirname, hydra_dir)
                os.makedirs(dirname, exist_ok=True)
                for file in files:
                    if file in selected_files[i]:
                        file = os.path.join(dirname, os.path.basename(file))
                        with open(file, "w") as f:
                            f.write(texts[i])
                            f.flush()
                    else:
                        shutil.copy(file, dirname)
            state.hydra_overrides = hydra_overrides
            state.train = train
    else:
        title(os.path.join(ROOT_DIR, "..", "images", "logo.png"))
        st.write("The agent is training...")
        st.write(
            "Please wait until it is done. For training information you can check out the **TB LOGS** tab or directly look at {}".format(
                urljoin(f"{APP_SERVER_HOST}:{APP_SERVER_PORT}", "view/" + quote("TB logs"))
            )
        )


class EditConfUI(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train = False
        self.hydra_overrides = None
        self.tmp_hydra_dir = tempfile.mkdtemp()

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render)


if __name__ == "__main__":
    render()
