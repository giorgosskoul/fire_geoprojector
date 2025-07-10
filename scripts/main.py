from utils.helper import simulate_fire_detections
import json

def main():
    # Load the test ignition points from JSON file
    with open('data/test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    ignition_points = data.get('fire_coords', [])
    if not ignition_points:
        print("No ignition points found in the JSON file.")
        return
    else:
        print(f"Loaded {len(ignition_points)} ignition points from the JSON file.")

    # Simulate fire detections with default parameters
    time_grid = simulate_fire_detections()
    
    # Print the simulated fire detections
    for lat, lon, timestamp in time_grid:
        print(f"Latitude: {lat}, Longitude: {lon}, Timestamp: {timestamp}")

if __name__ == "__main__":
    main()