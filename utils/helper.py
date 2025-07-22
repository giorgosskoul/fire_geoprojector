import json
import logging
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from utils.buffer import EventBuffer

LOGGER = logging.getLogger(__name__)


def load_points_from_json(file_path: str):
    """
    Load ignition points from a json file.
    Points must be written as a list of lists of two floats;
    latitude and longitude.

    :param file_path: Path to the JSON file containing ignition points.
    :return: List of ignition points, each represented as a tuple (lat, lon).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        ignition_points = json.load(f)
    if not ignition_points:
        LOGGER.info("No ignition points found in the JSON file.")
        return
    else:
        LOGGER.info(
            f"Loaded {len(ignition_points)} ignition points from the JSON file."
        )
        return ignition_points


def visualize_fire_grid(buffer: EventBuffer, gif_path: str = None):
    """
    Visualize the fire events stored in the EventBuffer tensor.
    A possible current limitation is that previous cells on fire are not shown in future
    detections.

    :param buffer: An instance of the EventBuffer.
    :param gif_path: Optional path to save the animation as a GIF.
    """
    frames = []
    for frame_idx in range(buffer.max_capacity):
        fire_layer = buffer.tensor[frame_idx, 1]
        # Skip empty frames
        if np.sum(fire_layer) == 0:
            continue
        frames.append(fire_layer.copy())

    if not frames:
        LOGGER.info("No fire events to visualize.")
        return

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.set_title("Fire Spread Over Time")
        ax.imshow(frame, cmap="hot", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)

    if gif_path:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        ani.save(gif_path, writer="pillow")
        LOGGER.info(f"Saved fire animation to {gif_path}")
    else:
        plt.show()

    plt.close(fig)


class LiveFireDisplay:
    def __init__(self, row_size, record=False):
        self.record = record
        self.frames = []
        self.row_size = row_size

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(
            np.zeros((row_size, row_size)), cmap="hot", vmin=0, vmax=1
        )
        self.ax.set_title("Live Fire Grid")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def update(self, fire_frame, title=None):
        self.img.set_data(fire_frame)
        if title:
            self.ax.set_title(title)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.record:
            self.frames.append(fire_frame.copy())

    def save_gif(self, path):
        if not self.record or not self.frames:
            LOGGER.info("No recorded frames to save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ani = animation.ArtistAnimation(
            self.fig,
            [[self.ax.imshow(f, cmap="hot", animated=True)] for f in self.frames],
            interval=500,
            blit=True,
        )
        ani.save(path, writer="pillow")
        LOGGER.info(f"Saved animation to {path}")
