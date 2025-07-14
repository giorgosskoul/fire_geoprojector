import argparse
from datetime import datetime

import numpy as np

from utils.buffer import EventBuffer
from utils.helper import (
    assign_timestamps_to_points,
    load_points_from_json,
    visualize_fire_grid,
)


def main(data_path: str, gif_path: str):
    """
    Main function to load fire detections, insert into buffer, and visualize the results.

    :param data_path: Path to the ignition points JSON file.
    :param gif_path: Path to save the output GIF. If None, the grid will be printed to the console.
    """
    # Load the test ignition points from JSON file
    ignition_points = load_points_from_json(data_path)
    if not ignition_points:
        print("No ignition points to process.")
        return

    print(f"Ignition Points: {ignition_points}")

    # Define timeline start
    start_time = datetime.now()
    timestamped_events = assign_timestamps_to_points(
        ignition_points, start_time, step_minutes=1
    )

    # Initialize EventBuffer
    ignition_point = timestamped_events[0]
    buffer = EventBuffer(ignition_point=ignition_point)

    # Add all events to buffer
    print("Adding events to buffer...")
    for lat, lon, ts in timestamped_events:
        buffer.add_event(lat, lon, ts)

    # Convert buffer tensor to (row, col, timestamp) detections for visualization
    detections = []
    for frame_idx in range(buffer.max_capacity):
        frame = buffer.tensor[frame_idx, 1]
        ts_frame = buffer.tensor[frame_idx, 0]
        rows, cols = np.where(frame > 0)
        for r, c in zip(rows, cols):
            ts_unix = ts_frame[r, c]
            timestamp = datetime.fromtimestamp(float(ts_unix))
            detections.append((r, c, timestamp))

    visualize_fire_grid(buffer, gif_path=gif_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Geoprojector Script")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/test.json",
        help="Path to the ignition points JSON file",
    )
    parser.add_argument(
        "--gif_path", type=str, default=None, help="Path to save the output GIF"
    )
    args = parser.parse_args()
    main(args.data_path, args.gif_path)
