import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def save_episode_as_gif(
    frames, path="./", episode_counter=0, action_probs: Optional[List[np.ndarray]] = None, dpi: int = 72
):
    """
    Saves a list of frames as a gif using matplotlib.
    Args:
        frames (list): List of frames to save as a gif.
        path (str): Path to save the gif.
        filename (str): Name of the gif.
        action_probs (list, optional): List of action probabilities. Defaults to None.
        dpi (int, optional): DPI of the gif. Defaults to 72.
    """
    num_columns = 1 if action_probs is None else len(action_probs[0]) + 1
    fig_size = (frames[0].shape[1] * num_columns / dpi, frames[0].shape[0] / dpi)
    fig, axes = plt.subplots(
        1,
        num_columns,
        figsize=fig_size,
        dpi=dpi,
        squeeze=False,
    )
    fig.suptitle(f"Episode {episode_counter}")
    axes[0][0].set_axis_off()
    img = axes[0][0].imshow(frames[0])
    bars = []
    if action_probs is not None:
        for j in range(len(action_probs[0])):

            bar_labels = [f"{idx}" for idx in range(len(action_probs[0][j]))]
            bar = axes[0][j + 1].bar(bar_labels, action_probs[0][j].tolist())
            axes[0][j + 1].set_ylim([0, 1.0])
            bars.append(bar)

    def animate(i):

        img.set_data(frames[i])
        axes[0][0].set_title(f"Step {i}")
        if action_probs is not None:
            for j in range(len(action_probs[i])):
                for k in range(len(action_probs[i][j])):
                    bars[j][k].set_height(action_probs[i][j][k])

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=33)
    anim.save(os.path.join(path, f"episode_{episode_counter}.gif"), writer="pillow", fps=60)
