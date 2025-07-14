from datetime import datetime, timedelta

import numpy as np


class EventBuffer:
    def __init__(
        self,
        ignition_point: tuple[float, float, datetime],
        row_size: int = 10,
        cell_size_km: float = 0.2,
        step_minutes: int = 1,
        max_capacity: int = 240,
    ):
        """
        A buffer class which stores fire detection events in a sequential grid.

        :param ignition_point: The fire-starting point corresponding to a tuple of (lat, lon, timestamp)
        :param row_size: Size of rows (also for columns). The total grid size will be row_size x row_size.
        :param cell_size_km: Size of each cell in km.
        :param step_minutes: Time between frames.
        :param max_capacity: The number of frames to retain.
        """
        self.lat0, self.lon0, self.start_time = ignition_point
        self.row_size = row_size
        self.cell_size_km = cell_size_km
        self.step_minutes = step_minutes
        self.max_capacity = max_capacity

        self.tensor = np.zeros((max_capacity, 12, row_size, row_size), dtype=np.float32)
        # (!): for the tensor I will use tensor[:,0,row,col] as the timestamp
        # # and tensor[:,1,row,col] as the true fire. Rest can be added later.

    def coord_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        """
        Converts real-world coordinates to grid coordinates.

        :param lat: Latitude of the point.
        :param lon: Longitude of the point.
        :return: Tuple of (row, col) corresponding to the grid position.
        """
        # We can approximate 1 degree of latitude to 111km, for small distances.
        # The real conversion requires the Haversine Formula.
        # We can then calculate 1 degree of longitutde as 111km * cos(lat).
        km_per_deg_lat = 111.320
        km_per_deg_lon = 111.320 * np.cos(np.radians(self.lat0))

        dlat_km = (lat - self.lat0) * km_per_deg_lat
        dlon_km = (lon - self.lon0) * km_per_deg_lon

        # Calculate offsets for grid rows and columns
        row_offset = int(round(dlat_km / self.cell_size_km))
        col_offset = int(round(dlon_km / self.cell_size_km))

        # Grid origin is at center
        center = self.row_size // 2
        # Latitude decreases as we're going south (down)
        row = center - row_offset
        # Longitude increases as we're going east (right)
        col = center + col_offset

        return row, col

    def get_frame_index(self, timestamp: datetime) -> int:
        """Get the index of the frame relative to the start time"""
        delta = timestamp - self.start_time
        return int(delta.total_seconds() // (self.step_minutes * 60))

    def add_event(self, lat: float, lon: float, timestamp: datetime):
        """
        Add a fire event to the buffer.

        :param lat: Latitude of the fire detection.
        :param lon: Longitude of the fire detection.
        :param timestamp: Datetime instance corresponding to the detection's time.
        """
        frame_idx = self.get_frame_index(timestamp)

        if not (0 <= frame_idx < self.max_capacity):
            return

        row, col = self.coord_to_grid(lat, lon)

        if 0 <= row < self.row_size and 0 <= col < self.row_size:
            # Store fire flag
            self.tensor[frame_idx, 1, row, col] = 1

            # Store timestamp (UNIX time) in channel 0 for the entire frame
            ts_unix = timestamp.timestamp()
            self.tensor[frame_idx, 0, row, col] = ts_unix
