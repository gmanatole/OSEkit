"""Auxiliary data represent values scattered through different auxiliary files."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import pandas as pd

from osekit.core.auxiliary_file import AuxiliaryFile
from osekit.core.auxiliary_item import AuxiliaryItem
from osekit.core.base_data import BaseData

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timestamp


_METADATA_COLUMNS = {
    "aggregation",
    "gps_source",
    "latitude",
    "longitude",
    "gps_latitude",
    "gps_longitude",
    "n_files",
    "n_rows",
    "source_files",
    "timestamp",
    "window_begin",
    "window_end",
}


class AuxiliaryData(BaseData[AuxiliaryItem, AuxiliaryFile]):
    """Auxiliary data represent rows retrieved from one or more auxiliary files."""

    item_cls = AuxiliaryItem

    def __init__(
        self,
        items: list[AuxiliaryItem] | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        aggregation: Literal["mean", "median"] = "mean",
        timestamp_position: Literal["begin", "center", "end"] = "begin",
        value_columns: Sequence[str] | None = None,
    ) -> None:
        """Initialize an ``AuxiliaryData``."""
        super().__init__(items=items, begin=begin, end=end, name=name)
        self.aggregation = aggregation
        self.timestamp_position = timestamp_position
        self.value_columns = value_columns

    @property
    def aggregation(self) -> Literal["mean", "median"]:
        """Aggregation method applied to the raw auxiliary rows."""
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value: Literal["mean", "median"]) -> None:
        if value not in {"mean", "median"}:
            msg = f"Unsupported aggregation method: {value}"
            raise ValueError(msg)
        self._aggregation = value

    @property
    def timestamp_position(self) -> Literal["begin", "center", "end"]:
        """Timestamp used when exporting the aggregated row."""
        return self._timestamp_position

    @timestamp_position.setter
    def timestamp_position(self, value: Literal["begin", "center", "end"]) -> None:
        if value not in {"begin", "center", "end"}:
            msg = f"Unsupported timestamp position: {value}"
            raise ValueError(msg)
        self._timestamp_position = value

    @property
    def value_columns(self) -> list[str]:
        """Variables written in the exported CSV row."""
        return self._value_columns

    @value_columns.setter
    def value_columns(self, columns: Sequence[str] | None) -> None:
        if columns is None:
            self._value_columns = self._infer_value_columns()
            return
        self._value_columns = list(dict.fromkeys(columns))

    def _infer_value_columns(self) -> list[str]:
        columns: list[str] = []
        for item in self.items:
            if item.file is None:
                continue
            for column in item.file.value_columns:
                if column not in columns and column not in _METADATA_COLUMNS:
                    columns.append(column)
        return columns

    @classmethod
    def _make_item(
        cls,
        file: AuxiliaryFile | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> AuxiliaryItem:
        return AuxiliaryItem(file=file, begin=begin, end=end)

    @classmethod
    def _make_file(cls, path: Path, begin: Timestamp) -> AuxiliaryFile:
        return AuxiliaryFile(path=path, begin=begin)

    def _make_split_data(
        self,
        files: list[AuxiliaryFile],
        begin: Timestamp,
        end: Timestamp,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        return AuxiliaryData.from_files(
            files=files,
            begin=begin,
            end=end,
            name=kwargs.get("name"),
            aggregation=self.aggregation,
            timestamp_position=self.timestamp_position,
            value_columns=self.value_columns,
        )

    def _combined_frame(self) -> pd.DataFrame:
        frames = [item.get_value() for item in self.items if not item.is_empty]
        if not frames:
            return pd.DataFrame(columns=self.value_columns)

        frame = pd.concat(frames, axis=0, sort=False).sort_index()
        if self.value_columns:
            for column in self.value_columns:
                if column not in frame.columns:
                    frame[column] = np.nan
            frame = frame.loc[:, self.value_columns]
        return frame

    def _aggregate_frame(self, frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series({column: np.nan for column in self.value_columns})

        output: dict[str, object] = {}
        for column in frame.columns:
            series = frame[column].dropna()
            if series.empty:
                output[column] = np.nan
                continue
            if pd.api.types.is_numeric_dtype(series):
                output[column] = getattr(series, self.aggregation)()
                continue
            values = list(pd.unique(series))
            output[column] = (
                values[0]
                if len(values) == 1
                else ";".join(str(value) for value in values)
            )
        return pd.Series(output)

    def _window_timestamp(self) -> Timestamp:
        if self.timestamp_position == "begin":
            return self.begin
        if self.timestamp_position == "end":
            return self.end
        return self.begin + self.duration / 2

    def to_frame(self) -> pd.DataFrame:
        """Return the aggregated auxiliary row as a one-row ``DataFrame``."""
        frame = self._combined_frame()
        values = self._aggregate_frame(frame)
        gps_track = next(
            (file.gps_track for file in self.files if file.gps_track is not None),
            None,
        )
        gps_coordinates = (
            None
            if gps_track is None
            else gps_track.coordinates_at([self._window_timestamp()]).iloc[0]
        )
        row = {
            "timestamp": self._window_timestamp(),
            "window_begin": self.begin,
            "window_end": self.end,
            "aggregation": self.aggregation,
            "n_rows": len(frame),
            "n_files": len(self.files),
            "source_files": ";".join(sorted(file.path.name for file in self.files)),
        }
        if gps_coordinates is not None:
            row.update(
                {
                    "gps_latitude": gps_coordinates["latitude"],
                    "gps_longitude": gps_coordinates["longitude"],
                    "gps_source": gps_track.label if gps_track is not None else None,
                },
            )
        row.update(values.to_dict())
        ordered_columns = [
            "timestamp",
            "window_begin",
            "window_end",
            "aggregation",
            "n_rows",
            "n_files",
            "source_files",
            "gps_latitude",
            "gps_longitude",
            "gps_source",
            *self.value_columns,
        ]
        if gps_coordinates is None:
            ordered_columns = [
                column
                for column in ordered_columns
                if column not in {"gps_latitude", "gps_longitude", "gps_source"}
            ]
        return pd.DataFrame([row], columns=ordered_columns)

    def get_value(self) -> pd.DataFrame:
        """Return the exported auxiliary row."""
        return self.to_frame()

    def write(
        self,
        folder: Path,
        *,
        link: bool = True,
    ) -> None:
        """Write the auxiliary row to a CSV file."""
        super().create_directories(path=folder)
        self.to_frame().to_csv(folder / f"{self}.csv", index=False)
        if link:
            self.link(folder=folder)

    def link(self, folder: Path) -> None:
        """Link the ``AuxiliaryData`` to the written CSV file."""
        file = AuxiliaryFile(path=folder / f"{self}.csv", begin=self.begin, end=self.end)
        self.items = AuxiliaryData.from_files(
            files=[file],
            begin=self.begin,
            end=self.end,
            aggregation=self.aggregation,
            timestamp_position=self.timestamp_position,
            value_columns=self.value_columns,
        ).items

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AuxiliaryData):
            return False
        return (
            self.aggregation == other.aggregation
            and self.timestamp_position == other.timestamp_position
            and self.value_columns == other.value_columns
            and super().__eq__(other)
        )

    def to_dict(self) -> dict:
        """Serialize an ``AuxiliaryData`` to a dictionary."""
        base_dict = super().to_dict()
        return base_dict | {
            "aggregation": self.aggregation,
            "timestamp_position": self.timestamp_position,
            "value_columns": self.value_columns,
        }

    @classmethod
    def from_dict(
        cls,
        dictionary: dict,
    ) -> Self:
        """Deserialize an ``AuxiliaryData`` from a dictionary."""
        files = [
            AuxiliaryFile.from_dict(file)
            for file in dictionary["files"].values()
        ]
        begin = pd.Timestamp(dictionary["begin"])
        end = pd.Timestamp(dictionary["end"])
        return cls._from_base_dict(
            dictionary=dictionary,
            files=files,
            begin=begin,
            end=end,
        )

    @classmethod
    def _from_base_dict(
        cls,
        dictionary: dict,
        files: list[AuxiliaryFile],
        begin: Timestamp,
        end: Timestamp,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        return cls.from_files(
            files=files,
            begin=begin,
            end=end,
            name=dictionary.get("name"),
            aggregation=dictionary.get("aggregation", "mean"),
            timestamp_position=dictionary.get("timestamp_position", "begin"),
            value_columns=dictionary.get("value_columns"),
        )

    @classmethod
    def from_files(
        cls,
        files: list[AuxiliaryFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        """Return an ``AuxiliaryData`` from a list of ``AuxiliaryFiles``."""
        return super().from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
            **kwargs,
        )
