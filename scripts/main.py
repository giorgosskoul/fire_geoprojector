from utils.helper import (
    load_points_from_json,
    simulate_fire_detections,
    visualize_fire_grid,
)
import argparse


def main(data_path: str, gif_path: str):
    """
    Main function to simulate fire detections, and visualize the results.

    :param data_path: Path to the ignition points JSON file.
    :param gif_path: Path to save the output GIF. If None, the grid will be printed to the console.
    """
    # Load the test ignition points from JSON file
    ignition_points = load_points_from_json(data_path)
    print(f"Ignition Points: {ignition_points}")

    # Simulate fire detections with default parameters
    detections = simulate_fire_detections()

    visualize_fire_grid(detections, gif_path=gif_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Geoprojector Script")
    parser.add_argument("--data_path", type=str, default="data/test.json", help="Path to the ignition points JSON file")
    parser.add_argument("--gif_path", type=str, default=None, help="Path to save the output GIF")
    args = parser.parse_args()
    main(args.data_path, args.gif_path)
