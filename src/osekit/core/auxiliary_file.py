"""Auxiliary file associated with timestamps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from pandas import Timestamp

from osekit.config import TIMESTAMP_FORMATS_EXPORTED_FILES
from osekit.core import auxiliary_file_manager as afm
from osekit.core.base_file import BaseFile
from osekit.core.gps_track import GpsTrack
from osekit.utils.timestamp import localize_timestamp, strptime_from_text

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    import pytz


class AuxiliaryFile(BaseFile):
    """Auxiliary file associated with timestamps."""

    supported_extensions: ClassVar = [
        ".csv",
        ".cdf",
        ".h5",
        ".hdf",
        ".hdf5",
        ".nc",
        ".nc4",
        ".netcdf",
    ]

    def __init__(
        self,
        path: PathLike | str,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        *,
        time_column: str | None = None,
        value_columns: Sequence[str] | None = None,
        wind_speed_column: str | None = None,
        hdf_key: str | None = None,
        time_unit: str | None = None,
        time_origin: str | Timestamp = "unix",
        timezone: str | pytz.timezone | None = None,
        gps_track: GpsTrack | None = None,
    ) -> None:
        """Initialize an ``AuxiliaryFile`` object.

        The begin and end timestamps are inferred from the file content if they are
        not provided explicitly. The file can contain either a CSV table, a HDF table
        or a NetCDF time series.
        """
        info = afm.info(
            path=path,
            time_column=time_column,
            value_columns=value_columns,
            wind_speed_column=wind_speed_column,
            hdf_key=hdf_key,
            time_unit=time_unit,
            time_origin=time_origin,
            timezone=timezone,
        )

        inferred_begin = info.begin if begin is None else begin
        inferred_end = info.end if end is None else end

        if timezone is not None:
            inferred_begin = localize_timestamp(inferred_begin, timezone)
            inferred_end = localize_timestamp(inferred_end, timezone)

        super().__init__(
            path=path,
            begin=inferred_begin,
            end=inferred_end,
        )

        self.time_column = info.time_column if time_column is None else time_column
        self.value_columns = list(info.value_columns)
        self.wind_speed_column = wind_speed_column
        self.latitude_column = info.latitude_column
        self.longitude_column = info.longitude_column
        self.gps_track = gps_track
        self.hdf_key = hdf_key
        self.time_unit = time_unit
        self.time_origin = time_origin
        self.timezone = timezone
        self.n_rows = info.n_rows

    @property
    def has_spatial_coordinates(self) -> bool:
        """Return ``True`` if the file contains latitude and longitude columns."""
        return self.latitude_column is not None and self.longitude_column is not None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AuxiliaryFile):
            return False
        return (
            super().__eq__(other)
            and self.time_column == other.time_column
            and self.value_columns == other.value_columns
            and self.wind_speed_column == other.wind_speed_column
            and self.hdf_key == other.hdf_key
            and self.time_unit == other.time_unit
            and str(self.time_origin) == str(other.time_origin)
            and str(self.timezone) == str(other.timezone)
            and self.n_rows == other.n_rows
            and self.latitude_column == other.latitude_column
            and self.longitude_column == other.longitude_column
            and self.gps_track == other.gps_track
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.path,
                self.begin,
                self.end,
                self.time_column,
                tuple(self.value_columns),
                self.wind_speed_column,
                self.hdf_key,
                self.time_unit,
                str(self.time_origin),
                str(self.timezone),
                self.n_rows,
                self.latitude_column,
                self.longitude_column,
                self.gps_track,
            ),
        )

    def read(self, start: Timestamp, stop: Timestamp) -> pd.DataFrame:
        """Return the auxiliary data between ``start`` and ``stop``."""
        return afm.read(
            path=self.path,
            start=start,
            stop=stop,
            time_column=self.time_column,
            value_columns=self.value_columns,
            wind_speed_column=self.wind_speed_column,
            hdf_key=self.hdf_key,
            time_unit=self.time_unit,
            time_origin=self.time_origin,
            timezone=self.timezone,
            gps_track=self.gps_track,
        )

    def move(self, folder: Path) -> None:
        """Move the file to the target folder."""
        afm.close()
        super().move(folder)

    def to_dict(self) -> dict:
        """Serialize the ``AuxiliaryFile`` to a dictionary."""
        serialized = super().to_dict()
        serialized.update(
            {
                "time_column": self.time_column,
                "value_columns": self.value_columns,
                "wind_speed_column": self.wind_speed_column,
                "hdf_key": self.hdf_key,
                "time_unit": self.time_unit,
                "time_origin": str(self.time_origin),
                "timezone": str(self.timezone) if self.timezone is not None else None,
                "n_rows": self.n_rows,
                "latitude_column": self.latitude_column,
                "longitude_column": self.longitude_column,
                "gps_track": None if self.gps_track is None else self.gps_track.to_dict(),
            },
        )
        return serialized

    @classmethod
    def from_dict(cls, serialized: dict) -> AuxiliaryFile:
        """Deserialize an ``AuxiliaryFile`` from a dictionary."""
        return cls(
            path=serialized["path"],
            begin=strptime_from_text(
                text=serialized["begin"],
                datetime_template=TIMESTAMP_FORMATS_EXPORTED_FILES,
            ),
            end=strptime_from_text(
                text=serialized["end"],
                datetime_template=TIMESTAMP_FORMATS_EXPORTED_FILES,
            ),
            time_column=serialized.get("time_column"),
            value_columns=serialized.get("value_columns"),
            wind_speed_column=serialized.get("wind_speed_column"),
            hdf_key=serialized.get("hdf_key"),
            time_unit=serialized.get("time_unit"),
            time_origin=serialized.get("time_origin", "unix"),
            timezone=serialized.get("timezone"),
            gps_track=GpsTrack.from_dict(serialized.get("gps_track")),
        )
