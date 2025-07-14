import json
import os
import random
from datetime import datetime, timedelta

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from utils.buffer import EventBuffer


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
        print("No ignition points found in the JSON file.")
        return
    else:
        print(f"Loaded {len(ignition_points)} ignition points from the JSON file.")
        return ignition_points


def assign_timestamps_to_points(
    points: list[list[float]],
    start_time: datetime,
    step_minutes: int = 1,
) -> list[tuple[float, float, datetime]]:
    """
    Assign timestamps to a list of (lat, lon) points in sequential order.

    :param points: List of [lat, lon] points.
    :param start_time: Datetime to begin the timeline.
    :param step_minutes: Minutes between each point.
    :return: List of (lat, lon, timestamp) tuples.
    """
    return [
        (lat, lon, start_time + timedelta(minutes=i * step_minutes))
        for i, (lat, lon) in enumerate(points)
    ]


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
        print("No fire events to visualize.")
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
        print(f"Saved fire animation to {gif_path}")
    else:
        plt.show()

    plt.close(fig)
