"""Helpers for loading auxiliary time series data from supported file formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from osekit.core.gps_track import GpsTrack

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas import Timestamp


_METADATA_COLUMNS = {
    "aggregation",
    "end",
    "gps_source",
    "gps_latitude",
    "gps_longitude",
    "latitude",
    "longitude",
    "n_files",
    "n_rows",
    "source_files",
    "start",
    "timestamp",
    "window_begin",
    "window_end",
}

_NETCDF_TIME_UNIT_MAP = {
    "seconds": "s",
    "second": "s",
    "sec": "s",
    "s": "s",
    "minutes": "m",
    "minute": "m",
    "min": "m",
    "hours": "h",
    "hour": "h",
    "h": "h",
    "days": "D",
    "day": "D",
    "d": "D",
    "milliseconds": "ms",
    "millisecond": "ms",
    "ms": "ms",
    "microseconds": "us",
    "microsecond": "us",
    "us": "us",
    "nanoseconds": "ns",
    "nanosecond": "ns",
    "ns": "ns",
}


@dataclass(frozen=True, slots=True)
class AuxiliaryFileInfo:
    """Basic metadata extracted from a supported auxiliary file."""

    begin: Timestamp
    end: Timestamp
    time_column: str
    value_columns: tuple[str, ...]
    n_rows: int
    latitude_column: str | None = None
    longitude_column: str | None = None

    @property
    def has_spatial_coordinates(self) -> bool:
        """Return ``True`` if the file stores latitude and longitude columns."""
        return self.latitude_column is not None and self.longitude_column is not None


class AuxiliaryFileManager:
    """Load and cache auxiliary time series frames."""

    def __init__(self) -> None:
        self._frame_cache: dict[tuple, pd.DataFrame] = {}

    def close(self) -> None:
        """Clear cached auxiliary frames."""
        self._frame_cache.clear()

    def info(
        self,
        path: Path | str,
        *,
        time_column: str | None = None,
        value_columns: Sequence[str] | None = None,
        wind_speed_column: str | None = None,
        hdf_key: str | None = None,
        time_unit: str | None = None,
        time_origin: str | Timestamp = "unix",
        timezone: str | None = None,
    ) -> AuxiliaryFileInfo:
        """Return metadata for a supported auxiliary file."""
        frame = self._standardize_frame(
            path=path,
            time_column=time_column,
            value_columns=value_columns,
            wind_speed_column=wind_speed_column,
            hdf_key=hdf_key,
            time_unit=time_unit,
            time_origin=time_origin,
            timezone=timezone,
        )
        if frame.empty:
            msg = f"No valid time series row found in {path}."
            raise ValueError(msg)

        selected_value_columns = self._infer_value_columns(
            frame=frame,
            time_column=frame.index.name or time_column,
            value_columns=value_columns,
        )
        latitude_column, longitude_column = self._spatial_columns(frame)
        return AuxiliaryFileInfo(
            begin=frame.index.min(),
            end=self._infer_end(frame.index),
            time_column=frame.index.name or (time_column or "timestamp"),
            value_columns=tuple(selected_value_columns),
            n_rows=len(frame),
            latitude_column=latitude_column,
            longitude_column=longitude_column,
        )

    def read(
        self,
        path: Path | str,
        start: Timestamp,
        stop: Timestamp,
        *,
        time_column: str | None = None,
        value_columns: Sequence[str] | None = None,
        wind_speed_column: str | None = None,
        hdf_key: str | None = None,
        time_unit: str | None = None,
        time_origin: str | Timestamp = "unix",
        timezone: str | None = None,
        gps_track: GpsTrack | None = None,
    ) -> pd.DataFrame:
        """Return the rows from ``path`` between ``start`` and ``stop``."""
        frame = self._standardize_frame(
            path=path,
            time_column=time_column,
            value_columns=value_columns,
            wind_speed_column=wind_speed_column,
            hdf_key=hdf_key,
            time_unit=time_unit,
            time_origin=time_origin,
            timezone=timezone,
        )
        if frame.empty:
            return pd.DataFrame(columns=self._infer_value_columns(
                frame=frame,
                time_column=time_column or frame.index.name,
                value_columns=value_columns,
            ))

        start, stop = self._align_timestamp_bounds(
            index=frame.index,
            start=start,
            stop=stop,
        )
        mask = (frame.index >= start) & (frame.index < stop)
        frame = frame.loc[mask].copy()
        if frame.empty:
            return pd.DataFrame(
                columns=self._infer_value_columns(
                    frame=frame,
                    time_column=time_column or frame.index.name,
                    value_columns=value_columns,
                ),
            )

        selected_value_columns = self._infer_value_columns(
            frame=frame,
            time_column=time_column or frame.index.name,
            value_columns=value_columns,
        )
        if not self._spatial_columns(frame)[0]:
            return frame.loc[:, selected_value_columns].copy()

        if gps_track is None or not gps_track.is_valid:
            msg = (
                "Auxiliary data contains latitude/longitude columns but no GPS "
                "coordinates were provided. Please input GPS coordinates or "
                "provide a GPS CSV file."
            )
            raise ValueError(msg)

        sampled = self._sample_spatial_frame(
            frame=frame,
            gps_track=gps_track,
            value_columns=selected_value_columns,
        )
        return sampled.loc[:, selected_value_columns].copy()

    def _align_timestamp_bounds(
        self,
        *,
        index: pd.DatetimeIndex,
        start: Timestamp,
        stop: Timestamp,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Normalize the slice bounds to the timezone of ``index``."""
        start_ts = pd.Timestamp(start)
        stop_ts = pd.Timestamp(stop)
        index_tz = index.tz

        if index_tz is None:
            if start_ts.tz is not None:
                start_ts = start_ts.tz_localize(None)
            if stop_ts.tz is not None:
                stop_ts = stop_ts.tz_localize(None)
            return start_ts, stop_ts

        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(index_tz)
        else:
            start_ts = start_ts.tz_convert(index_tz)

        if stop_ts.tz is None:
            stop_ts = stop_ts.tz_localize(index_tz)
        else:
            stop_ts = stop_ts.tz_convert(index_tz)

        return start_ts, stop_ts

    def _standardize_frame(
        self,
        path: Path | str,
        *,
        time_column: str | None,
        value_columns: Sequence[str] | None,
        hdf_key: str | None,
        time_unit: str | None,
        time_origin: str | Timestamp,
        timezone: str | None,
        wind_speed_column: str | None,
    ) -> pd.DataFrame:
        cache_key = (
            str(Path(path).resolve()),
            time_column,
            tuple(value_columns) if value_columns is not None else None,
            wind_speed_column,
            hdf_key,
            time_unit,
            str(time_origin),
            timezone,
        )
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key].copy()

        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            frame = self._load_csv(Path(path))
        elif suffix in {".h5", ".hdf", ".hdf5"}:
            frame = self._load_hdf(Path(path), hdf_key=hdf_key)
        elif suffix in {".nc", ".nc4", ".cdf", ".netcdf"}:
            frame = self._load_netcdf(Path(path))
        else:
            msg = f"Unsupported auxiliary file extension: {suffix}"
            raise ValueError(msg)

        netcdf_units = frame.attrs.get("_netcdf_units", {})
        frame = self._normalize_raw_frame(
            frame=frame,
            time_column=time_column,
            value_columns=value_columns,
            time_unit=time_unit,
            time_origin=time_origin,
            timezone=timezone,
            source_path=Path(path),
            netcdf_units=netcdf_units,
        )
        self._frame_cache[cache_key] = frame
        return frame.copy()

    def _load_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def _load_hdf(self, path: Path, *, hdf_key: str | None) -> pd.DataFrame:
        try:
            if hdf_key is None:
                with pd.HDFStore(path, mode="r") as store:
                    keys = store.keys()
                if not keys:
                    msg = f"No table found in {path}."
                    raise ValueError(msg)
                hdf_key = keys[0]
            frame = pd.read_hdf(path, key=hdf_key)
        except ImportError as exc:
            msg = (
                "HDF support requires the optional pandas HDF dependency "
                "(PyTables / `tables`)."
            )
            raise ImportError(msg) from exc

        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        return frame

    def _load_netcdf(self, path: Path) -> pd.DataFrame:
        try:
            from xarray import open_dataset
        except ImportError:
            return self._load_netcdf_scipy(path)
        with open_dataset(path) as dataset:
            frame = dataset.to_dataframe().reset_index()
        return frame

    def _load_netcdf_scipy(self, path: Path) -> pd.DataFrame:
        from scipy.io import netcdf_file

        with netcdf_file(path, mode="r", mmap=False) as dataset:
            variables: dict[str, np.ndarray] = {}
            units: dict[str, str] = {}
            for name, variable in dataset.variables.items():
                array = np.array(variable[:])
                if array.dtype.byteorder not in ("=", "|"):
                    array = array.byteswap().view(array.dtype.newbyteorder("="))
                variables[name] = array
                unit = getattr(variable, "units", None)
                if unit:
                    if isinstance(unit, bytes):
                        unit = unit.decode()
                    units[name] = str(unit)
        frame = pd.DataFrame(variables)
        frame.attrs["_netcdf_units"] = units
        return frame

    def _normalize_raw_frame(
        self,
        *,
        frame: pd.DataFrame,
        time_column: str | None,
        value_columns: Sequence[str] | None,
        time_unit: str | None,
        time_origin: str | Timestamp,
        timezone: str | None,
        source_path: Path,
        netcdf_units: dict[str, str] | None,
    ) -> pd.DataFrame:
        frame = frame.copy()
        frame = self._normalize_coordinate_columns(frame)

        inferred_time_column = self._infer_time_column(
            frame=frame,
            time_column=time_column,
        )
        time_values, frame = self._extract_time_values(
            frame=frame,
            time_column=inferred_time_column,
            time_unit=time_unit,
            time_origin=time_origin,
            timezone=timezone,
            source_path=source_path,
            netcdf_units=netcdf_units,
        )
        frame.index = time_values
        frame.index.name = inferred_time_column or "timestamp"
        frame = frame.sort_index()
        return frame

    def _normalize_coordinate_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = set(frame.columns)
        if {"lat", "lon"} <= columns:
            if {"latitude", "longitude"} & columns:
                msg = (
                    "Auxiliary data must not contain both 'lat'/'lon' and "
                    "'latitude'/'longitude' columns."
                )
                raise ValueError(msg)
            return frame.rename(columns={"lat": "latitude", "lon": "longitude"})
        return frame

    def _infer_time_column(
        self,
        *,
        frame: pd.DataFrame,
        time_column: str | None,
    ) -> str | None:
        if time_column is not None:
            if time_column in frame.columns:
                return time_column
            if frame.index.name == time_column:
                return time_column
            msg = f"Time column '{time_column}' was not found."
            raise ValueError(msg)

        if isinstance(frame.index, pd.DatetimeIndex):
            return frame.index.name or "timestamp"

        for candidate in ("timestamp", "time", "datetime", "date"):
            if candidate in frame.columns:
                return candidate

        if len(frame.columns) == 1:
            return frame.columns[0]

        return None

    def _infer_value_columns(
        self,
        *,
        frame: pd.DataFrame,
        time_column: str | None,
        value_columns: Sequence[str] | None,
    ) -> list[str]:
        if value_columns is not None:
            if time_column is not None and time_column in value_columns:
                msg = "The time column cannot be part of the selected value columns."
                raise ValueError(msg)
            missing = [
                column
                for column in value_columns
                if column not in frame.columns
            ]
            if missing:
                msg = f"Missing auxiliary columns: {', '.join(missing)}."
                raise ValueError(msg)
            selected = [
                column
                for column in value_columns
                if column not in _METADATA_COLUMNS
            ]
            if not selected:
                msg = "No non-metadata auxiliary columns were selected."
                raise ValueError(msg)
            return selected

        excluded = set(_METADATA_COLUMNS)
        if time_column is not None:
            excluded.add(time_column)

        columns = [column for column in frame.columns if column not in excluded]
        if columns:
            return columns

        return [
            column
            for column in frame.columns
            if column != time_column and column not in _METADATA_COLUMNS
        ]

    def _extract_time_values(
        self,
        *,
        frame: pd.DataFrame,
        time_column: str | None,
        time_unit: str | None,
        time_origin: str | Timestamp,
        timezone: str | None,
        source_path: Path,
        netcdf_units: dict[str, str] | None,
    ) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
        if isinstance(frame.index, pd.DatetimeIndex) and time_column in (None, frame.index.name):
            index = frame.index
            if timezone is not None:
                index = self._normalize_timezone(index=index, timezone=timezone)
            return index, frame

        if time_column is None:
            msg = f"No time column could be inferred for {source_path}."
            raise ValueError(msg)

        if time_column in frame.columns:
            time_values = frame[time_column]
        elif frame.index.name == time_column:
            time_values = frame.index
        else:
            msg = f"Time column '{time_column}' was not found in {source_path}."
            raise ValueError(msg)

        if time_unit is None and netcdf_units and time_column in netcdf_units:
            decoded = self._decode_netcdf_time_units(netcdf_units[time_column])
            if decoded is not None:
                time_unit, time_origin = decoded

        time_index = self._coerce_to_datetime_index(
            values=time_values,
            time_unit=time_unit,
            time_origin=time_origin,
            timezone=timezone,
        )
        if isinstance(time_values, pd.Series) and time_column in frame.columns:
            frame = frame.drop(columns=[time_column])
        elif frame.index.name == time_column:
            frame = frame.reset_index(drop=True)
        return time_index, frame

    def _coerce_to_datetime_index(
        self,
        *,
        values: pd.Series | pd.Index | np.ndarray,
        time_unit: str | None,
        time_origin: str | Timestamp,
        timezone: str | None,
    ) -> pd.DatetimeIndex:
        if time_unit is not None:
            timestamps = pd.to_datetime(
                values,
                unit=time_unit,
                origin=time_origin,
                errors="raise",
            )
        else:
            timestamps = pd.to_datetime(values, errors="raise")

        if isinstance(timestamps, pd.Series):
            timestamps = pd.DatetimeIndex(timestamps)
        elif not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)

        if timezone is not None:
            timestamps = self._normalize_timezone(index=timestamps, timezone=timezone)
        return timestamps

    def _spatial_columns(
        self,
        frame: pd.DataFrame,
    ) -> tuple[str | None, str | None]:
        columns = set(frame.columns)
        if {"latitude", "longitude"} <= columns:
            return "latitude", "longitude"
        return None, None

    def _sample_spatial_frame(
        self,
        *,
        frame: pd.DataFrame,
        gps_track: GpsTrack,
        value_columns: Sequence[str],
    ) -> pd.DataFrame:
        query_times = pd.DatetimeIndex(sorted(pd.unique(frame.index)))
        if query_times.empty:
            output = pd.DataFrame(columns=list(value_columns), index=query_times)
            output.index.name = frame.index.name or "timestamp"
            return output

        query_coordinates = gps_track.coordinates_at(query_times).reindex(query_times)
        try:
            return self._sample_with_regular_grid(
                frame=frame,
                query_times=query_times,
                query_coordinates=query_coordinates,
                value_columns=value_columns,
            )
        except Exception:
            return self._sample_with_nearest(
                frame=frame,
                query_times=query_times,
                query_coordinates=query_coordinates,
                value_columns=value_columns,
            )

    def _sample_with_regular_grid(
        self,
        *,
        frame: pd.DataFrame,
        query_times: pd.DatetimeIndex,
        query_coordinates: pd.DataFrame,
        value_columns: Sequence[str],
    ) -> pd.DataFrame:
        working = frame.reset_index().rename(
            columns={frame.index.name or "index": "timestamp"},
        )
        required_columns = {"timestamp", "latitude", "longitude"}
        if not required_columns <= set(working.columns):
            msg = "Auxiliary spatial data must contain timestamp, latitude and longitude."
            raise ValueError(msg)

        working = working.loc[:, ["timestamp", "latitude", "longitude", *value_columns]]
        working = working.drop_duplicates(
            subset=["timestamp", "latitude", "longitude"],
            keep="last",
        )

        unique_times = pd.DatetimeIndex(sorted(pd.unique(working["timestamp"])))
        unique_latitudes = np.asarray(
            pd.to_numeric(pd.unique(working["latitude"]), errors="raise"),
            dtype=float,
        )
        unique_longitudes = np.asarray(
            pd.to_numeric(pd.unique(working["longitude"]), errors="raise"),
            dtype=float,
        )
        unique_latitudes.sort()
        unique_longitudes.sort()

        expected_rows = (
            len(unique_times) * len(unique_latitudes) * len(unique_longitudes)
        )
        if len(working) != expected_rows:
            msg = "Auxiliary data is not on a complete regular grid."
            raise ValueError(msg)

        grid_index = pd.MultiIndex.from_product(
            [unique_times, unique_latitudes, unique_longitudes],
            names=["timestamp", "latitude", "longitude"],
        )
        grid = (
            working.set_index(["timestamp", "latitude", "longitude"])
            .loc[:, list(value_columns)]
            .reindex(grid_index)
        )
        if grid.isna().any().any():
            msg = "Auxiliary regular-grid interpolation would require missing values."
            raise ValueError(msg)

        origin = unique_times[0]
        time_axis = self._datetime_to_seconds(unique_times, origin)
        query_time_axis = self._datetime_to_seconds(query_times, origin)
        query_points = np.column_stack(
            [
                query_time_axis,
                pd.to_numeric(query_coordinates["latitude"], errors="raise").to_numpy(
                    dtype=float,
                ),
                pd.to_numeric(query_coordinates["longitude"], errors="raise").to_numpy(
                    dtype=float,
                ),
            ],
        )

        output = pd.DataFrame(index=query_times)
        for column in value_columns:
            values = grid.loc[:, column].to_numpy().reshape(
                len(unique_times),
                len(unique_latitudes),
                len(unique_longitudes),
            )
            interpolator = RegularGridInterpolator(
                (time_axis, unique_latitudes, unique_longitudes),
                values,
                bounds_error=False,
                fill_value=np.nan,
            )
            sampled = np.asarray(interpolator(query_points), dtype=float)
            if np.isnan(sampled).any():
                msg = "Auxiliary regular-grid interpolation returned missing values."
                raise ValueError(msg)
            output[column] = sampled

        output.index.name = frame.index.name or "timestamp"
        return output

    def _sample_with_nearest(
        self,
        *,
        frame: pd.DataFrame,
        query_times: pd.DatetimeIndex,
        query_coordinates: pd.DataFrame,
        value_columns: Sequence[str],
    ) -> pd.DataFrame:
        working = frame.reset_index().rename(
            columns={frame.index.name or "index": "timestamp"},
        )
        source_times = pd.DatetimeIndex(sorted(pd.unique(working["timestamp"])))
        if source_times.empty:
            output = pd.DataFrame(columns=list(value_columns), index=query_times)
            output.index.name = frame.index.name or "timestamp"
            return output

        output_rows: list[dict[str, object]] = []
        for query_time in query_times:
            nearest_time = self._nearest_timestamp(query_time, source_times)
            time_slice = working.loc[working["timestamp"] == nearest_time]
            if time_slice.empty:
                continue

            query_latitude = float(query_coordinates.loc[query_time, "latitude"])
            query_longitude = float(query_coordinates.loc[query_time, "longitude"])
            distances = self._haversine_distance(
                latitude=time_slice["latitude"].to_numpy(dtype=float),
                longitude=time_slice["longitude"].to_numpy(dtype=float),
                query_latitude=query_latitude,
                query_longitude=query_longitude,
            )
            nearest_row = time_slice.iloc[int(np.nanargmin(distances))]
            output_row = {
                "timestamp": query_time,
                **{
                    column: nearest_row[column]
                    for column in value_columns
                },
            }
            output_rows.append(output_row)

        if not output_rows:
            output = pd.DataFrame(columns=list(value_columns), index=query_times)
            output.index.name = frame.index.name or "timestamp"
            return output

        output = pd.DataFrame(output_rows).set_index("timestamp")
        output.index.name = frame.index.name or "timestamp"
        return output.loc[:, list(value_columns)]

    @staticmethod
    def _nearest_timestamp(
        timestamp: Timestamp,
        source_times: pd.DatetimeIndex,
    ) -> Timestamp:
        deltas = np.abs(source_times.asi8 - pd.Timestamp(timestamp).value)
        return source_times[int(np.argmin(deltas))]

    @staticmethod
    def _datetime_to_seconds(
        timestamps: pd.DatetimeIndex,
        origin: Timestamp,
    ) -> np.ndarray:
        origin_value = pd.Timestamp(origin).value
        return (timestamps.asi8.astype(np.float64) - float(origin_value)) / 1_000_000_000

    @staticmethod
    def _haversine_distance(
        *,
        latitude: np.ndarray,
        longitude: np.ndarray,
        query_latitude: float,
        query_longitude: float,
    ) -> np.ndarray:
        earth_radius_m = 6_371_000.0
        latitudes = np.radians(latitude)
        longitudes = np.radians(longitude)
        query_latitude_rad = np.radians(query_latitude)
        query_longitude_rad = np.radians(query_longitude)
        d_lat = latitudes - query_latitude_rad
        d_lon = longitudes - query_longitude_rad
        a = (
            np.sin(d_lat / 2) ** 2
            + np.cos(latitudes)
            * np.cos(query_latitude_rad)
            * np.sin(d_lon / 2) ** 2
        )
        return 2 * earth_radius_m * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def _normalize_timezone(
        self,
        *,
        index: pd.DatetimeIndex,
        timezone: str,
    ) -> pd.DatetimeIndex:
        if index.tz is None:
            return index.tz_localize(timezone)
        return index.tz_convert(timezone)

    def _infer_end(self, index: pd.DatetimeIndex) -> Timestamp:
        if len(index) <= 1:
            return index.max() + pd.Timedelta(seconds=1)

        deltas = index.to_series().diff().dropna()
        positive_deltas = deltas[deltas > pd.Timedelta(0)]
        step = (
            positive_deltas.median()
            if not positive_deltas.empty
            else pd.Timedelta(seconds=1)
        )
        return index.max() + step

    def _decode_netcdf_time_units(self, units: str) -> tuple[str, str] | None:
        match = re.match(
            r"^(?P<unit>[A-Za-z]+)\s+since\s+(?P<origin>.+)$",
            units.strip(),
        )
        if match is None:
            return None
        unit = match.group("unit").lower()
        pandas_unit = _NETCDF_TIME_UNIT_MAP.get(unit)
        if pandas_unit is None:
            return None
        return pandas_unit, match.group("origin").strip()
