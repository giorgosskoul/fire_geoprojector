import random
from datetime import datetime, timedelta

def simulate_fire_detections(
    num_frames: int = 4,
    pattern: str = "line",
    step_minutes: float = 30.0,
    lat_range: tuple[float] = (38.004, 38.014),
    lon_range: tuple[float] =( 23.956, 23.966),
):
    """
    Simulate dummy fire detections.

    The patterns of fire spread can be:
    - 'line': Fire spreads in a line.
    - 'random': Fire spreads randomly within the specified latitude and longitude range.
    - 'center_spread': Fire spreads from the center of the specified range circularly.

    :parm num_frames: Number of frames to simulate.
    :parm pattern: Pattern of fire spread ('line', 'random', 'center_spread').
    :parm step_minutes: Time step in minutes between frames.
    :parm lat_range: Tuple of (min_lat, max_lat) for latitude.
    :parm lon_range: Tuple of (min_lon, max_lon) for longitude.
    :return: List of tuples containing (latitude, longitude, timestamp) for each frame.
    """
    
    start_time = datetime.now()

    detections = []

    for i in range(num_frames):
        timestamp = start_time + timedelta(minutes=i * step_minutes)

        if pattern == "line":
            # Fire spreads eastward (longitude increases)
            lat = (lat_range[0] + lat_range[1]) / 2
            lon = lon_range[0] + (i / max(num_frames - 1, 1)) * (lon_range[1] - lon_range[0])
            detections.append((lat, lon, timestamp))

        elif pattern == "random":
            lat = random.uniform(*lat_range)
            lon = random.uniform(*lon_range)
            detections.append((lat, lon, timestamp))

        elif pattern == "center_spread":
            center_lat = (lat_range[0] + lat_range[1]) / 2
            center_lon = (lon_range[0] + lon_range[1]) / 2
            # Spread out in a circular way
            lat_offset = random.uniform(-0.001, 0.001) * (i + 1)
            lon_offset = random.uniform(-0.001, 0.001) * (i + 1)
            detections.append((center_lat + lat_offset, center_lon + lon_offset, timestamp))

        else:
            raise ValueError(f"Unsupported pattern: {pattern}")

    return detections
