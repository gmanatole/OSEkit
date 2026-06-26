"""Helpers for GPS tracks used to spatially join auxiliary data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

_TIME_CANDIDATES = ("timestamp", "timestamps", "time", "datetime", "date")


def _normalize_timezone(
    index: pd.DatetimeIndex,
    timezone: str,
) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(timezone)
    return index.tz_convert(timezone)


def _infer_time_column(columns: pd.Index) -> str:
    for candidate in _TIME_CANDIDATES:
        if candidate in columns:
            return candidate
    msg = (
        "GPS CSV must contain a timestamp column such as "
        "'timestamp', 'time', 'datetime', or 'date'."
    )
    raise ValueError(msg)


def _normalize_coordinate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = set(frame.columns)
    if {"lat", "lon"} <= columns:
        if {"latitude", "longitude"} & columns:
            msg = (
                "GPS CSV must not contain both 'lat'/'lon' and "
                "'latitude'/'longitude' columns."
            )
            raise ValueError(msg)
        return frame.rename(columns={"lat": "latitude", "lon": "longitude"})
    if {"latitude", "longitude"} <= columns:
        return frame
    msg = (
        "GPS CSV must contain either 'lat'/'lon' or "
        "'latitude'/'longitude' columns."
    )
    raise ValueError(msg)


@dataclass(slots=True)
class GpsTrack:
    """GPS information used to spatially sample auxiliary time series."""

    path: Path | None = None
    latitude: float | None = None
    longitude: float | None = None
    time_column: str = "timestamp"
    latitude_column: str = "latitude"
    longitude_column: str = "longitude"
    timezone: str | None = None
    _frame_cache: pd.DataFrame | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    @classmethod
    def fixed(
        cls,
        latitude: float,
        longitude: float,
        *,
        timezone: str | None = None,
    ) -> Self:
        """Return a track defined by fixed coordinates."""
        return cls(
            path=None,
            latitude=float(latitude),
            longitude=float(longitude),
            timezone=timezone,
        )

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        timezone: str | None = None,
    ) -> Self:
        """Return a GPS track loaded from a CSV file."""
        path = Path(path)
        if not path.exists():
            msg = f"GPS CSV file not found: {path}"
            raise FileNotFoundError(msg)

        header = pd.read_csv(path, nrows=0)
        time_column = _infer_time_column(header.columns)
        _ = _normalize_coordinate_columns(header)
        return cls(
            path=path,
            time_column=time_column,
            latitude_column="latitude",
            longitude_column="longitude",
            timezone=timezone,
        )

    @property
    def kind(self) -> Literal["fixed", "mobile"]:
        """Return the GPS track kind."""
        return "mobile" if self.path is not None else "fixed"

    @property
    def label(self) -> str:
        """Return a short human-readable label."""
        return "fixed" if self.path is None else self.path.name

    @property
    def is_valid(self) -> bool:
        """Return ``True`` if the track can be used for spatial joins."""
        if self.kind == "mobile":
            return self.path is not None and self.path.exists()
        return (
            self.latitude is not None
            and self.longitude is not None
            and not (self.latitude == 0 and self.longitude == 0)
        )

    def _load_frame(self) -> pd.DataFrame:
        if self.kind == "fixed":
            msg = "Fixed GPS tracks do not have a tabular frame."
            raise ValueError(msg)
        if self._frame_cache is not None:
            return self._frame_cache.copy()

        assert self.path is not None  # for type checkers
        frame = pd.read_csv(self.path)
        frame = _normalize_coordinate_columns(frame)

        if self.time_column not in frame.columns:
            self.time_column = _infer_time_column(frame.columns)

        frame = frame.loc[:, [self.time_column, "latitude", "longitude"]].copy()
        timestamps = pd.to_datetime(frame[self.time_column], errors="raise")
        if isinstance(timestamps, pd.Series):
            timestamps = pd.DatetimeIndex(timestamps)
        elif not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
        if self.timezone is not None:
            timestamps = _normalize_timezone(timestamps, self.timezone)
        frame[self.time_column] = timestamps
        frame = (
            frame.sort_values(self.time_column)
            .drop_duplicates(subset=[self.time_column], keep="last")
            .set_index(self.time_column)
        )
        frame.index.name = self.time_column
        self._frame_cache = frame
        return frame.copy()

    def _align_query_index(self, timestamps: Sequence[pd.Timestamp]) -> pd.DatetimeIndex:
        index = pd.DatetimeIndex(timestamps)
        if self.kind == "fixed":
            return index

        frame = self._load_frame()
        frame_tz = frame.index.tz
        if frame_tz is None:
            if index.tz is not None:
                return index.tz_localize(None)
            return index

        if index.tz is None:
            return index.tz_localize(frame_tz)
        return index.tz_convert(frame_tz)

    def coordinates_at(
        self,
        timestamps: Sequence[pd.Timestamp],
    ) -> pd.DataFrame:
        """Return the GPS coordinates for the requested timestamps."""
        query_index = self._align_query_index(timestamps)
        if query_index.empty:
            return pd.DataFrame(
                columns=[self.latitude_column, self.longitude_column],
                index=query_index,
            )

        if self.kind == "fixed":
            frame = pd.DataFrame(
                {
                    self.latitude_column: [self.latitude] * len(query_index),
                    self.longitude_column: [self.longitude] * len(query_index),
                },
                index=query_index,
            )
            frame.index.name = self.time_column
            return frame

        frame = self._load_frame().reset_index()
        query = pd.DataFrame({self.time_column: query_index})
        aligned = pd.merge_asof(
            query.sort_values(self.time_column),
            frame.loc[:, [self.time_column, "latitude", "longitude"]].sort_values(
                self.time_column,
            ),
            on=self.time_column,
            direction="nearest",
        )
        aligned = aligned.set_index(self.time_column)
        aligned.index.name = self.time_column
        return aligned.loc[:, ["latitude", "longitude"]]

    def to_dict(self) -> dict:
        """Serialize the GPS track to a dictionary."""
        if self.kind == "fixed":
            return {
                "kind": "fixed",
                "latitude": self.latitude,
                "longitude": self.longitude,
                "timezone": self.timezone,
            }
        return {
            "kind": "mobile",
            "path": str(self.path),
            "time_column": self.time_column,
            "latitude_column": self.latitude_column,
            "longitude_column": self.longitude_column,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, serialized: dict | None) -> GpsTrack | None:
        """Deserialize a GPS track from a dictionary."""
        if serialized is None:
            return None
        kind = serialized.get("kind")
        if kind == "fixed":
            return cls.fixed(
                latitude=float(serialized["latitude"]),
                longitude=float(serialized["longitude"]),
                timezone=serialized.get("timezone"),
            )
        if kind == "mobile":
            return cls(
                path=Path(serialized["path"]),
                time_column=serialized.get("time_column", "timestamp"),
                latitude_column=serialized.get("latitude_column", "latitude"),
                longitude_column=serialized.get("longitude_column", "longitude"),
                timezone=serialized.get("timezone"),
            )
        msg = f"Unsupported GPS track kind: {kind}"
        raise ValueError(msg)

    def __hash__(self) -> int:
        return hash(
            (
                self.path,
                self.latitude,
                self.longitude,
                self.time_column,
                self.latitude_column,
                self.longitude_column,
                self.timezone,
            ),
        )


def find_mobile_gps_csv(folder: Path) -> Path | None:
    """Return the GPS CSV stored alongside an audio dataset, if any."""
    auxiliary_folder = folder / "auxiliary"
    if not auxiliary_folder.exists():
        return None

    candidates = [
        path
        for path in auxiliary_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    ]
    if not candidates:
        return None

    def is_gps_candidate(path: Path) -> bool:
        header = pd.read_csv(path, nrows=0)
        columns = set(header.columns)
        has_time = any(candidate in columns for candidate in _TIME_CANDIDATES)
        has_coordinates = {"latitude", "longitude"} <= columns or {"lat", "lon"} <= columns
        return has_time and has_coordinates

    matching = [path for path in candidates if is_gps_candidate(path)]
    if len(matching) == 1:
        return matching[0]

    gps_named = [path for path in matching if "gps" in path.stem.lower()]
    if len(gps_named) == 1:
        return gps_named[0]

    if len(matching) > 1:
        msg = (
            "Multiple GPS CSV files were found in the auxiliary folder. "
            "Please pass the GPS CSV explicitly."
        )
        raise ValueError(msg)
    return None


def resolve_gps_track(
    gps_coordinates: str | Sequence[float] | Path | GpsTrack | None,
    *,
    dataset_folder: Path | None = None,
    timezone: str | None = None,
) -> GpsTrack | None:
    """Resolve a GPS track from project-style coordinates or a CSV path."""
    if gps_coordinates is None:
        return None
    if isinstance(gps_coordinates, GpsTrack):
        return gps_coordinates
    if isinstance(gps_coordinates, Path):
        if gps_coordinates.suffix.lower() == ".csv" or gps_coordinates.exists():
            return GpsTrack.from_path(gps_coordinates, timezone=timezone)
        msg = "GPS path must point to an existing CSV file."
        raise ValueError(msg)
    if isinstance(gps_coordinates, str):
        if gps_coordinates == "mobile":
            if dataset_folder is None:
                return None
            gps_file = find_mobile_gps_csv(dataset_folder)
            return None if gps_file is None else GpsTrack.from_path(gps_file, timezone=timezone)
        candidate = Path(gps_coordinates)
        if candidate.suffix.lower() == ".csv" or candidate.exists():
            return GpsTrack.from_path(candidate, timezone=timezone)
        return None
    if len(gps_coordinates) != 2:
        msg = "GPS coordinates must contain exactly two values: latitude and longitude."
        raise ValueError(msg)
    latitude, longitude = gps_coordinates
    return GpsTrack.fixed(latitude=float(latitude), longitude=float(longitude), timezone=timezone)
