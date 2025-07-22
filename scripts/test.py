import argparse
import logging
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from rich.logging import RichHandler

from utils.buffer import EventBuffer
from utils.helper import LiveFireDisplay, load_points_from_json

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)],
)

LOGGER = logging.getLogger(__name__)


def main(data_path: str, gif_path: str):
    ignition_points = load_points_from_json(data_path)
    if not ignition_points:
        LOGGER.info("No ignition points to process.")
        return

    start_time = datetime.now()
    first_lat, first_lon = ignition_points[0][0]
    ignition_point = (first_lat, first_lon, start_time)
    buffer = EventBuffer(ignition_point=ignition_point)

    # Create live display with optional recording
    live_display = LiveFireDisplay(buffer.row_size, record=(gif_path is not None))

    LOGGER.info("Adding events and updating live view...")
    for t, fire_points in enumerate(ignition_points):
        ts = start_time + timedelta(minutes=t)
        for point in fire_points:
            if len(point) != 2:
                continue
            lat, lon = point
            buffer.add_event(lat, lon, ts)

        # Get the updated fire frame
        frame_idx = buffer.get_frame_index(ts)
        fire_frame = buffer.tensor[frame_idx, 1]
        live_display.update(fire_frame, title=f"Frame {frame_idx}")

        # Optional: pause to simulate real-time
        time.sleep(0.3)

    plt.ioff()
    plt.show()

    # 👇 save gif if needed
    if gif_path:
        live_display.save_gif(gif_path)


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
