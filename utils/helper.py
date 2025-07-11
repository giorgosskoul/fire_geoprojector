import random
from datetime import datetime, timedelta
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_points_from_json(file_path: str):
    """
    Load ignition points from a json file.
    Points must be written as a list of lists of two floats;
    latitude and longitude.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        ignition_points = json.load(f)
    if not ignition_points:
        print("No ignition points found in the JSON file.")
        return
    else:
        print(f"Loaded {len(ignition_points)} ignition points from the JSON file.")
        return ignition_points


def simulate_fire_detections(
    num_frames: int = 6,
    step_minutes: float = 10.0,
    grid_size: int = 10,
):
    """
    Simulate dummy fire detections on a 10x10 grid.
    The fire starts at the center of the grid and spreads randomly.

    :parm num_frames: Number of frames to simulate.
    :parm step_minutes: Time step in minutes between frames.
    :parm grid_size: Size of the grid (default 10 for 10x10).
    :return: List of tuples containing (row, col, timestamp) for each frame.
    """
    start_time = datetime.now()
    detections = []

    center = grid_size // 2
    row = center
    col = center

    # Initial detection at the center
    detections.append((row, col, start_time))
    ## !: For an even grid size (inc. 10), the center cannot be a tuple of integers.
    ## # Therefore, we do not use the real center, but a neighboring cell.
    ## # eg. (5,5) should not be the real center, but (4.5, 4.5)

    for i in range(num_frames):
        timestamp = start_time + timedelta(minutes=i * step_minutes)

        # Fire spread -> Only follows a random direction
        chance = random.random()
        if random.random() < 0.25:
            row = row - 1
        elif chance < 0.5:
            row = row + 1
        elif chance < 0.75:
            col = col - 1
        else:
            col = col + 1

        # Add detection only if within grid bounds
        if 0 <= row < grid_size and 0 <= col < grid_size:
            detections.append((row, col, timestamp))

    return detections



def visualize_fire_grid(detections, grid_size: int = 10, gif_path: str = None):
        """
        Visualize the fire detections on a grid for each timestamp.
        If gif_path is not specified, prints the grid to the console.
        Otherwise, stores the visualization as a GIF.
        """
        # Try using emojis if supported
        try:
            tree = "🌲"
            fire = "🔥"
            print(tree, fire, end="\r")
        except Exception:
            tree = "."
            fire = "F"

        # Prepare frames for each timestamp
        frames = []
        for i in range(len(detections)):
            grid = [[tree for _ in range(grid_size)] for _ in range(grid_size)]
            for row, col, _ in detections[:i+1]:
                if 0 <= row < grid_size and 0 <= col < grid_size:
                    grid[row][col] = fire
            frames.append(grid)

        if gif_path:
            # Ensure gif_path exists
            gif_dir = os.path.dirname(gif_path)
            if gif_dir and not os.path.exists(gif_dir):
                os.makedirs(gif_dir, exist_ok=True)

            # Create the GIF
            fig, ax = plt.subplots()
            def update(frame):
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(np.array([[1 if cell == fire else 0 for cell in row] for row in frame]), cmap="hot", vmin=0, vmax=1)
            ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
            ani.save(gif_path, writer='pillow')
            plt.close(fig)
        else:
            # Print each frame to the console
            for idx, grid in enumerate(frames):
                print(f"Frame {idx+1}:")
                for r in grid:
                    print(" ".join(r))
                print()
