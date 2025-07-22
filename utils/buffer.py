import logging
from datetime import datetime

import numpy as np

LOGGER = logging.getLogger(__name__)


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
        A buffer class which stores fire detection events in a tensor containing the sequential grid.

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

        # We can approximate 1 degree of latitude to 111km, for small distances.
        # The real conversion requires the Haversine Formula.
        # We can then calculate 1 degree of longitutde as 111km * cos(lat).
        self.km_per_deg_lat = 111.320
        self.km_per_deg_lon = 111.320 * np.cos(np.radians(self.lat0))

        # Define the grid origin
        LOGGER.info(
            f"Initializing EventBuffer with origin at ({self.lat0}, {self.lon0})"
        )

    def coord_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        """
        Converts real-world coordinates to grid coordinates.

        :param lat: Latitude of the point.
        :param lon: Longitude of the point.
        :return: Tuple of (row, col) corresponding to the grid position.
        """

        dlat_km = (lat - self.lat0) * self.km_per_deg_lat
        dlon_km = (lon - self.lon0) * self.km_per_deg_lon

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
            # Store the fire flag
            self.tensor[frame_idx, 1, row, col] = 1

            # Store timestamp (UNIX time) in channel 0 for the entire frame
            ts_unix = timestamp.timestamp()
            self.tensor[frame_idx, 0, row, col] = ts_unix
        else:
            # Overwrite with expanded grid
            self.handle_ofg_events(lat, lon, timestamp)

    def handle_ofg_events(self, lat: float, lon: float, timestamp: datetime):
        """
        Handles Out-of-Fire-Grid (OFG) events by shifting the grid origin.
        The new origin is calculated as the average of the current origin and the OFG event coordinates.

        :param lat: Latitude of the OFG event.
        :param lon: Longitude of the OFG event.
        """

        # Calculate the new grid origin based on the OFG event
        new_lat0 = (self.lat0 + lat) / 2
        new_lon0 = (self.lon0 + lon) / 2

        # Display a warning for OFG events
        LOGGER.info(
            f"Shifting grid origin to ({new_lat0}, {new_lon0}) due to OFG event at ({lat}, {lon})"
        )

        # Shift the grid to the new origin
        self.shift_grid(new_lat0, new_lon0)

        # Add the event at the new origin
        self.add_event(lat, lon, timestamp)

    def shift_grid(self, new_lat0: float, new_lon0: float):
        """
        Shift the grid origin to a new latitude and longitude.

        :param new_lat0: New latitude for the grid origin.
        :param new_lon0: New longitude for the grid origin.
        """
        km_per_deg_lon_old = self.km_per_deg_lon
        # Calculate new km per degree for the new origin
        self.km_per_deg_lon = 111.320 * np.cos(np.radians(new_lat0))

        center = self.row_size // 2

        # Old grid (row, col) → lat/lon
        rows, cols = np.meshgrid(
            np.arange(self.row_size), np.arange(self.row_size), indexing="ij"
        )
        # Calculate the distance in km from the center
        dlat_km = (center - rows) * self.cell_size_km
        dlon_km = (cols - center) * self.cell_size_km

        lat = self.lat0 + dlat_km / self.km_per_deg_lat
        lon = self.lon0 + dlon_km / km_per_deg_lon_old

        # lat/lon → new (row, col)
        dlat_km_new = (lat - new_lat0) * self.km_per_deg_lat
        dlon_km_new = (lon - new_lon0) * self.km_per_deg_lon

        new_rows = center - np.round(dlat_km_new / self.cell_size_km).astype(int)
        new_cols = center + np.round(dlon_km_new / self.cell_size_km).astype(int)

        # Mask for valid new indices
        valid_mask = (
            (new_rows >= 0)
            & (new_rows < self.row_size)
            & (new_cols >= 0)
            & (new_cols < self.row_size)
        )

        # Index valid old and new positions
        old_rows = rows[valid_mask]
        old_cols = cols[valid_mask]
        tgt_rows = new_rows[valid_mask]
        tgt_cols = new_cols[valid_mask]

        # Broadcast across time and channels
        new_tensor = np.zeros_like(self.tensor)
        new_tensor[:, :, tgt_rows, tgt_cols] = self.tensor[:, :, old_rows, old_cols]

        # Final update
        self.lat0 = new_lat0
        self.lon0 = new_lon0
        self.tensor = new_tensor

    def get_tensor(self) -> np.ndarray:
        """
        Returns the current tensor.

        :return: A numpy array of shape (max_capacity, 12, row_size, row_size).
        """
        return self.tensor
