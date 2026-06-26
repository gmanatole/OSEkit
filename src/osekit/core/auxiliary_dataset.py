"""AuxiliaryDataset is a collection of ``AuxiliaryData`` objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from osekit.core.auxiliary_data import AuxiliaryData
from osekit.core.auxiliary_file import AuxiliaryFile
from osekit.core.base_dataset import BaseDataset
from osekit.core.gps_track import resolve_gps_track
from osekit.core.json_serializer import deserialize_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytz
    from pandas import Timestamp

    from osekit.core.audio_dataset import AudioDataset


class AuxiliaryDataset(BaseDataset[AuxiliaryData, AuxiliaryFile]):
    """Collection of auxiliary data rows aligned on an audio dataset."""

    file_cls = AuxiliaryFile

    def __init__(
        self,
        data: list[AuxiliaryData],
        name: str | None = None,
        suffix: str = "auxiliary",
        folder: Path | None = None,
        aggregation: Literal["mean", "median"] = "mean",
        timestamp_position: Literal["begin", "center", "end"] = "begin",
        value_columns: Sequence[str] | None = None,
    ) -> None:
        """Initialize an ``AuxiliaryDataset``."""
        super().__init__(data=data, name=name, suffix=suffix, folder=folder)
        self.aggregation = aggregation
        self.timestamp_position = timestamp_position
        self.value_columns = value_columns

    @property
    def aggregation(self) -> Literal["mean", "median"]:
        """Return the dataset aggregation method."""
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value: Literal["mean", "median"]) -> None:
        if value not in {"mean", "median"}:
            msg = f"Unsupported aggregation method: {value}"
            raise ValueError(msg)
        self._aggregation = value
        for data in self.data:
            data.aggregation = value

    @property
    def timestamp_position(self) -> Literal["begin", "center", "end"]:
        """Return the timestamp position used in exported rows."""
        return self._timestamp_position

    @timestamp_position.setter
    def timestamp_position(self, value: Literal["begin", "center", "end"]) -> None:
        if value not in {"begin", "center", "end"}:
            msg = f"Unsupported timestamp position: {value}"
            raise ValueError(msg)
        self._timestamp_position = value
        for data in self.data:
            data.timestamp_position = value

    @property
    def value_columns(self) -> list[str]:
        """Return the auxiliary variables exported by the dataset."""
        if self._value_columns is not None:
            return self._value_columns
        columns: list[str] = []
        for data in self.data:
            for column in data.value_columns:
                if column not in columns:
                    columns.append(column)
        return columns

    @value_columns.setter
    def value_columns(self, columns: Sequence[str] | None) -> None:
        self._value_columns = None if columns is None else list(dict.fromkeys(columns))
        for data in self.data:
            data.value_columns = self._value_columns

    @staticmethod
    def _normalize_value_columns(
        value_columns: Sequence[str] | None,
    ) -> list[str] | None:
        if value_columns is None:
            return None
        return list(dict.fromkeys(value_columns))

    def _infer_value_columns(self) -> list[str]:
        columns: list[str] = []
        for data in self.data:
            for column in data.value_columns:
                if column not in columns:
                    columns.append(column)
        return columns

    @classmethod
    def _data_from_dict(cls, dictionary: dict) -> list[AuxiliaryData]:
        data_objects = []
        for name, data in dictionary.items():
            auxiliary_data = AuxiliaryData.from_dict(data)
            auxiliary_data.name = name
            data_objects.append(auxiliary_data)
        return data_objects

    @classmethod
    def _data_from_files(
        cls,
        files: list[AuxiliaryFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> AuxiliaryData:
        return AuxiliaryData.from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
            **kwargs,
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
        """Return an ``AuxiliaryDataset`` from a list of auxiliary files."""
        return super().from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        *,
        name: str | None = None,
        aggregation: Literal["mean", "median"] = "mean",
        timestamp_position: Literal["begin", "center", "end"] = "begin",
        value_columns: Sequence[str] | None = None,
        wind_speed_column: str | None = None,
        time_column: str | None = None,
        hdf_key: str | None = None,
        time_unit: str | None = None,
        time_origin: str | Timestamp = "unix",
        timezone: str | pytz.timezone | None = None,
    ) -> Self:
        """Create an ``AuxiliaryDataset`` from a folder of supported files."""
        normalized_value_columns = cls._normalize_value_columns(value_columns)
        files = [
            AuxiliaryFile(
                path=file,
                time_column=time_column,
                value_columns=normalized_value_columns,
                wind_speed_column=wind_speed_column,
                hdf_key=hdf_key,
                time_unit=time_unit,
                time_origin=time_origin,
                timezone=timezone,
            )
            for file in sorted(folder.iterdir())
            if file.suffix.lower() in AuxiliaryFile.supported_extensions
        ]
        if not files:
            msg = f"No supported auxiliary file found in {folder}."
            raise FileNotFoundError(msg)
        return cls.from_files(
            files=files,
            name=name,
            aggregation=aggregation,
            timestamp_position=timestamp_position,
            value_columns=normalized_value_columns,
        )

    @classmethod
    def from_audio_dataset(
        cls,
        audio_dataset: AudioDataset,
        files: Sequence[Path | str | AuxiliaryFile],
        *,
        name: str | None = None,
        aggregation: Literal["mean", "median"] = "mean",
        timestamp_position: Literal["begin", "center", "end"] = "begin",
        value_columns: Sequence[str] | None = None,
        wind_speed_column: str | None = None,
        time_column: str | None = None,
        hdf_key: str | None = None,
        time_unit: str | None = None,
        time_origin: str | Timestamp = "unix",
        timezone: str | pytz.timezone | None = None,
        gps_coordinates: str | Sequence[float] | Path | None = None,
        folder: Path | None = None,
    ) -> Self:
        """Create an auxiliary dataset aligned on the windows of an audio dataset."""
        aligned_timezone = timezone or audio_dataset.begin.tz
        normalized_value_columns = cls._normalize_value_columns(value_columns)
        resolved_gps_track = resolve_gps_track(
            gps_coordinates
            if gps_coordinates is not None
            else getattr(audio_dataset, "gps_coordinates", None),
            dataset_folder=audio_dataset.folder,
            timezone=aligned_timezone,
        )

        def _align_file_timezone(file: AuxiliaryFile) -> None:
            current_timezone = file.begin.tz
            if aligned_timezone is None:
                if current_timezone is not None:
                    file.begin = file.begin.tz_localize(None)
                    file.end = file.end.tz_localize(None)
                return
            if current_timezone is None:
                file.begin = file.begin.tz_localize(aligned_timezone)
                file.end = file.end.tz_localize(aligned_timezone)
                return
            file.begin = file.begin.tz_convert(aligned_timezone)
            file.end = file.end.tz_convert(aligned_timezone)

        auxiliary_files: list[AuxiliaryFile] = []
        for file in files:
            if isinstance(file, AuxiliaryFile):
                if normalized_value_columns is not None:
                    file.value_columns = normalized_value_columns
                if wind_speed_column is not None:
                    file.wind_speed_column = wind_speed_column
                _align_file_timezone(file)
                auxiliary_files.append(file)
                continue
            auxiliary_file = AuxiliaryFile(
                path=file,
                time_column=time_column,
                value_columns=normalized_value_columns,
                wind_speed_column=wind_speed_column,
                hdf_key=hdf_key,
                time_unit=time_unit,
                time_origin=time_origin,
                timezone=aligned_timezone,
                gps_track=(
                    resolved_gps_track
                    if resolved_gps_track is not None and resolved_gps_track.is_valid
                    else None
                ),
            )
            _align_file_timezone(auxiliary_file)
            auxiliary_files.append(auxiliary_file)

        has_spatial_coordinates = any(
            file.has_spatial_coordinates for file in auxiliary_files
        )
        if has_spatial_coordinates:
            if resolved_gps_track is None or not resolved_gps_track.is_valid:
                msg = (
                    "Auxiliary data contains latitude/longitude columns but no GPS "
                    "coordinates were provided. Please input GPS coordinates or "
                    "provide a GPS CSV file."
                )
                raise ValueError(msg)
        elif resolved_gps_track is not None and resolved_gps_track.is_valid:
            for file in auxiliary_files:
                if file.gps_track is None or not file.gps_track.is_valid:
                    file.gps_track = resolved_gps_track
        else:
            for file in auxiliary_files:
                if file.gps_track is not None and not file.gps_track.is_valid:
                    file.gps_track = None

        dataset_name = audio_dataset.name if name is None else name
        output_folder = (
            folder
            if folder is not None
            else audio_dataset.folder / "auxiliary" / f"{dataset_name}_auxiliary"
        )

        data = [
            AuxiliaryData.from_files(
                files=auxiliary_files,
                begin=audio_data.begin,
                end=audio_data.end,
                name=audio_data.name,
                aggregation=aggregation,
                timestamp_position=timestamp_position,
                value_columns=normalized_value_columns,
            )
            for audio_data in audio_dataset.data
        ]
        return cls(
            data=data,
            name=dataset_name,
            folder=output_folder,
            aggregation=aggregation,
            timestamp_position=timestamp_position,
            value_columns=normalized_value_columns,
        )

    def write(
        self,
        folder: Path,
        first: int = 0,
        last: int | None = None,
        *,
        link: bool = True,
    ) -> None:
        """Write the auxiliary dataset to CSV files and serialize its JSON."""
        self.folder = folder
        super().write(folder=folder, first=first, last=last, link=link)
        self.write_json(folder=folder)

    def to_dict(self) -> dict:
        """Serialize the ``AuxiliaryDataset`` to a dictionary."""
        base_dict = super().to_dict()
        return base_dict | {
            "aggregation": self.aggregation,
            "timestamp_position": self.timestamp_position,
            "value_columns": self.value_columns,
        }

    @classmethod
    def from_dict(cls, dictionary: dict) -> Self:
        """Deserialize an ``AuxiliaryDataset`` from a dictionary."""
        return cls(
            data=cls._data_from_dict(dictionary["data"]),
            name=dictionary["name"],
            suffix=dictionary.get("suffix", "auxiliary"),
            folder=Path(dictionary["folder"]),
            aggregation=dictionary.get("aggregation", "mean"),
            timestamp_position=dictionary.get("timestamp_position", "begin"),
            value_columns=dictionary.get("value_columns"),
        )

    @classmethod
    def from_json(cls, file: Path) -> Self:
        """Deserialize an ``AuxiliaryDataset`` from a JSON file."""
        return cls.from_dict(deserialize_json(file))
