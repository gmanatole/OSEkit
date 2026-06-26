from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp
from scipy.io import netcdf_file

from osekit.core.audio_dataset import AudioDataset
from osekit.core.audio_file import AudioFile
from osekit.core.auxiliary_dataset import AuxiliaryDataset
from osekit.core.auxiliary_file import AuxiliaryFile


def _make_auxiliary_csv(path: Path) -> pd.DataFrame:
    timestamps = pd.date_range(
        start=Timestamp("2024-01-01 00:00:00"),
        periods=20,
        freq="100ms",
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": [
                0,
                0,
                0,
                0,
                10,
                10,
                10,
                10,
                10,
                10,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                100,
                100,
            ],
            "sensor": ["S1"] * 20,
        },
    )
    frame.to_csv(path, index=False)
    return frame


def _make_regular_grid_auxiliary_csv(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in pd.date_range(
        start=Timestamp("2024-01-01 00:00:00"),
        periods=2,
        freq="1s",
    ):
        for latitude in (48.0, 49.0):
            for longitude in (-5.0, -4.0):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "latitude": latitude,
                        "longitude": longitude,
                        "temperature": float(timestamp.second) + latitude + longitude,
                    },
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _make_sparse_auxiliary_csv(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in pd.date_range(
        start=Timestamp("2024-01-01 00:00:00"),
        periods=2,
        freq="1s",
    ):
        rows.extend(
            [
                {
                    "timestamp": timestamp,
                    "latitude": 48.0,
                    "longitude": -5.0,
                    "temperature": 10.0 + float(timestamp.second),
                },
                {
                    "timestamp": timestamp,
                    "latitude": 48.2,
                    "longitude": -4.8,
                    "temperature": 20.0 + float(timestamp.second),
                },
            ],
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _make_wind_speed_source_csv(path: Path) -> pd.DataFrame:
    timestamps = pd.date_range(
        start=Timestamp("2024-01-01 00:00:00"),
        periods=20,
        freq="100ms",
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "in-situ_wind": [5.0] * 10 + [9.0] * 10,
        },
    )
    frame.to_csv(path, index=False)
    return frame


def _make_u10_v10_auxiliary_csv(path: Path) -> pd.DataFrame:
    timestamps = pd.date_range(
        start=Timestamp("2024-01-01 00:00:00"),
        periods=20,
        freq="100ms",
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "u10": [3.0] * 10 + [6.0] * 10,
            "v10": [4.0] * 10 + [8.0] * 10,
        },
    )
    frame.to_csv(path, index=False)
    return frame


def _make_audio_dataset(
    audio_files: tuple[list, pytest.fixtures.Subrequest],
    *,
    tmp_path: Path,
    timezone: str | None = None,
) -> AudioDataset:
    files, _ = audio_files
    if timezone is not None:
        files = [
            AudioFile(
                path=audio_file.path,
                begin=audio_file.begin,
                timezone=timezone,
            )
            for audio_file in files
        ]
    audio_dataset = AudioDataset.from_files(
        files,
        name="original",
        mode="files",
    )
    audio_dataset.folder = tmp_path / "project" / "data" / "audio" / "original"
    return audio_dataset


SPATIAL_AUDIO_FILES = pytest.param(
    {
        "duration": 1.0,
        "sample_rate": 10,
        "nb_files": 2,
        "date_begin": pd.Timestamp("2024-01-01 00:00:00"),
        "series_type": "repeat",
    },
    id="two_audio_windows",
)


@pytest.mark.parametrize(
    ("aggregation", "expected_first", "expected_second"),
    [
        pytest.param("mean", 6.0, 21.2, id="mean"),
        pytest.param("median", 10.0, 2.0, id="median"),
    ],
)
@pytest.mark.parametrize(
    "audio_files",
    [
        pytest.param(
            {
                "duration": 1.0,
                "sample_rate": 10,
                "nb_files": 2,
                "date_begin": pd.Timestamp("2024-01-01 00:00:00"),
                "series_type": "repeat",
            },
            id="two_audio_windows",
        ),
    ],
    indirect=True,
)
def test_auxiliary_dataset_from_audio_dataset_and_write(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
    aggregation: str,
    expected_first: float,
    expected_second: float,
) -> None:
    files, _ = audio_files
    audio_dataset = AudioDataset.from_files(files, name="original", mode="files")
    audio_dataset.folder = tmp_path / "project" / "data" / "audio" / "original"

    csv_path = tmp_path / "auxiliary_source.csv"
    _ = _make_auxiliary_csv(csv_path)

    auxiliary_dataset = AuxiliaryDataset.from_audio_dataset(
        audio_dataset=audio_dataset,
        files=[csv_path],
        aggregation=aggregation,
    )

    output_folder = audio_dataset.folder / "auxiliary" / auxiliary_dataset.name
    auxiliary_dataset.write(output_folder)

    csv_files = sorted(output_folder.glob("*.csv"))
    assert len(csv_files) == 2
    assert (output_folder / f"{auxiliary_dataset.name}.json").exists()

    written = [pd.read_csv(file) for file in csv_files]
    assert [pd.Timestamp(frame.loc[0, "timestamp"]) for frame in written] == [
        audio_dataset.data[0].begin,
        audio_dataset.data[1].begin,
    ]

    assert np.isclose(written[0].loc[0, "temperature"], expected_first)
    assert np.isclose(written[1].loc[0, "temperature"], expected_second)
    assert all(frame.loc[0, "sensor"] == "S1" for frame in written)

    loaded = AuxiliaryDataset.from_json(output_folder / f"{auxiliary_dataset.name}.json")
    assert loaded == auxiliary_dataset


@pytest.mark.parametrize("audio_files", [SPATIAL_AUDIO_FILES], indirect=True)
def test_auxiliary_dataset_with_regular_grid_and_fixed_gps(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
) -> None:
    audio_dataset = _make_audio_dataset(audio_files, tmp_path=tmp_path)
    csv_path = tmp_path / "auxiliary_regular_grid.csv"
    _ = _make_regular_grid_auxiliary_csv(csv_path)

    auxiliary_dataset = AuxiliaryDataset.from_audio_dataset(
        audio_dataset=audio_dataset,
        files=[csv_path],
        aggregation="mean",
        gps_coordinates=(48.5, -4.5),
    )

    output_folder = audio_dataset.folder / "auxiliary" / auxiliary_dataset.name
    auxiliary_dataset.write(output_folder)

    csv_files = sorted(output_folder.glob("*.csv"))
    assert len(csv_files) == 2

    written = [pd.read_csv(file) for file in csv_files]
    assert pd.Timestamp(written[0].loc[0, "timestamp"]) == Timestamp(
        "2024-01-01 00:00:00",
    )
    assert pd.Timestamp(written[1].loc[0, "timestamp"]) == Timestamp(
        "2024-01-01 00:00:01",
    )
    assert written[0].loc[0, "temperature"] == pytest.approx(44.0)
    assert written[1].loc[0, "temperature"] == pytest.approx(45.0)
    assert written[0].loc[0, "gps_latitude"] == pytest.approx(48.5)
    assert written[0].loc[0, "gps_longitude"] == pytest.approx(-4.5)
    assert written[0].loc[0, "gps_source"] == "fixed"

    loaded = AuxiliaryDataset.from_json(output_folder / f"{auxiliary_dataset.name}.json")
    assert loaded == auxiliary_dataset


@pytest.mark.parametrize("audio_files", [SPATIAL_AUDIO_FILES], indirect=True)
def test_auxiliary_dataset_with_mobile_gps_csv(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
) -> None:
    audio_dataset = _make_audio_dataset(audio_files, tmp_path=tmp_path)
    audio_dataset.gps_coordinates = "mobile"

    gps_folder = audio_dataset.folder / "auxiliary"
    gps_folder.mkdir(parents=True, exist_ok=True)
    gps_csv = gps_folder / "gps.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range(
                Timestamp("2024-01-01 00:00:00"),
                periods=2,
                freq="1s",
            ),
            "lat": [48.5, 48.25],
            "lon": [-4.5, -4.75],
        },
    ).to_csv(gps_csv, index=False)

    csv_path = tmp_path / "auxiliary_regular_grid.csv"
    _ = _make_regular_grid_auxiliary_csv(csv_path)

    auxiliary_dataset = AuxiliaryDataset.from_audio_dataset(
        audio_dataset=audio_dataset,
        files=[csv_path],
        aggregation="mean",
    )

    output_folder = audio_dataset.folder / "auxiliary" / auxiliary_dataset.name
    auxiliary_dataset.write(output_folder)

    csv_files = sorted(output_folder.glob("*.csv"))
    assert len(csv_files) == 2

    written = [pd.read_csv(file) for file in csv_files]
    assert written[0].loc[0, "temperature"] == pytest.approx(44.0)
    assert written[1].loc[0, "temperature"] == pytest.approx(44.5)
    assert written[0].loc[0, "gps_latitude"] == pytest.approx(48.5)
    assert written[0].loc[0, "gps_longitude"] == pytest.approx(-4.5)
    assert written[0].loc[0, "gps_source"] == "gps.csv"


@pytest.mark.parametrize(
    "audio_files",
    [
        pytest.param(
            {
                "duration": 1.0,
                "sample_rate": 10,
                "nb_files": 2,
                "date_begin": pd.Timestamp("2024-01-01 00:00:00"),
                "series_type": "repeat",
            },
            id="two_audio_windows",
        ),
    ],
    indirect=True,
)
def test_auxiliary_dataset_preserves_selected_column_name(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
) -> None:
    audio_dataset = _make_audio_dataset(audio_files, tmp_path=tmp_path)
    csv_path = tmp_path / "auxiliary_wind_source.csv"
    _ = _make_wind_speed_source_csv(csv_path)

    auxiliary_dataset = AuxiliaryDataset.from_audio_dataset(
        audio_dataset=audio_dataset,
        files=[csv_path],
        aggregation="mean",
        value_columns=["in-situ_wind"],
    )

    assert auxiliary_dataset.value_columns == ["in-situ_wind"]

    output_folder = audio_dataset.folder / "auxiliary" / auxiliary_dataset.name
    auxiliary_dataset.write(output_folder)

    csv_files = sorted(output_folder.glob("*.csv"))
    assert len(csv_files) == 2

    written = [pd.read_csv(file) for file in csv_files]
    assert "in-situ_wind" in written[0].columns
    assert "wind_speed" not in written[0].columns
    assert written[0].loc[0, "in-situ_wind"] == pytest.approx(5.0)
    assert written[1].loc[0, "in-situ_wind"] == pytest.approx(9.0)

    loaded = AuxiliaryDataset.from_json(output_folder / f"{auxiliary_dataset.name}.json")
    assert loaded == auxiliary_dataset


@pytest.mark.parametrize(
    "audio_files",
    [
        pytest.param(
            {
                "duration": 1.0,
                "sample_rate": 10,
                "nb_files": 2,
                "date_begin": pd.Timestamp("2024-01-01 00:00:00"),
                "series_type": "repeat",
            },
            id="two_audio_windows",
        ),
    ],
    indirect=True,
)
def test_auxiliary_dataset_preserves_u10_and_v10_columns(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
) -> None:
    audio_dataset = _make_audio_dataset(audio_files, tmp_path=tmp_path)
    csv_path = tmp_path / "auxiliary_uv.csv"
    _ = _make_u10_v10_auxiliary_csv(csv_path)

    auxiliary_dataset = AuxiliaryDataset.from_audio_dataset(
        audio_dataset=audio_dataset,
        files=[csv_path],
        aggregation="mean",
        value_columns=["u10", "v10"],
    )

    assert auxiliary_dataset.value_columns == ["u10", "v10"]

    output_folder = audio_dataset.folder / "auxiliary" / auxiliary_dataset.name
    auxiliary_dataset.write(output_folder)

    csv_files = sorted(output_folder.glob("*.csv"))
    assert len(csv_files) == 2

    written = [pd.read_csv(file) for file in csv_files]
    assert list(written[0].columns) == [
        "timestamp",
        "window_begin",
        "window_end",
        "aggregation",
        "n_rows",
        "n_files",
        "source_files",
        "u10",
        "v10",
    ]
    assert written[0].loc[0, "u10"] == pytest.approx(3.0)
    assert written[1].loc[0, "u10"] == pytest.approx(6.0)
    assert written[0].loc[0, "v10"] == pytest.approx(4.0)
    assert written[1].loc[0, "v10"] == pytest.approx(8.0)

    loaded = AuxiliaryDataset.from_json(output_folder / f"{auxiliary_dataset.name}.json")
    assert loaded == auxiliary_dataset


@pytest.mark.parametrize("audio_files", [SPATIAL_AUDIO_FILES], indirect=True)
def test_auxiliary_dataset_with_sparse_grid_uses_nearest_fallback(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
) -> None:
    audio_dataset = _make_audio_dataset(audio_files, tmp_path=tmp_path)
    csv_path = tmp_path / "auxiliary_sparse.csv"
    _ = _make_sparse_auxiliary_csv(csv_path)

    auxiliary_dataset = AuxiliaryDataset.from_audio_dataset(
        audio_dataset=audio_dataset,
        files=[csv_path],
        aggregation="mean",
        gps_coordinates=(48.18, -4.82),
    )

    output_folder = audio_dataset.folder / "auxiliary" / auxiliary_dataset.name
    auxiliary_dataset.write(output_folder)

    csv_files = sorted(output_folder.glob("*.csv"))
    assert len(csv_files) == 2

    written = [pd.read_csv(file) for file in csv_files]
    assert written[0].loc[0, "temperature"] == pytest.approx(20.0)
    assert written[1].loc[0, "temperature"] == pytest.approx(21.0)
    assert written[0].loc[0, "gps_latitude"] == pytest.approx(48.18)
    assert written[0].loc[0, "gps_longitude"] == pytest.approx(-4.82)


@pytest.mark.parametrize("audio_files", [SPATIAL_AUDIO_FILES], indirect=True)
def test_auxiliary_dataset_with_spatial_data_requires_gps_coordinates(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
) -> None:
    audio_dataset = _make_audio_dataset(audio_files, tmp_path=tmp_path)
    csv_path = tmp_path / "auxiliary_sparse.csv"
    _ = _make_sparse_auxiliary_csv(csv_path)

    with pytest.raises(ValueError, match=r"GPS coordinates|GPS CSV"):
        AuxiliaryDataset.from_audio_dataset(
            audio_dataset=audio_dataset,
            files=[csv_path],
            aggregation="mean",
            gps_coordinates=(0, 0),
        )


@pytest.mark.parametrize(
    ("audio_timezone", "aux_timezone"),
    [
        pytest.param(None, "UTC", id="naive_audio_aware_aux"),
        pytest.param("UTC", None, id="aware_audio_naive_aux"),
    ],
)
@pytest.mark.parametrize("audio_files", [SPATIAL_AUDIO_FILES], indirect=True)
def test_auxiliary_dataset_temporal_join_handles_timezone_mismatch(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
    audio_timezone: str | None,
    aux_timezone: str | None,
) -> None:
    audio_dataset = _make_audio_dataset(
        audio_files,
        tmp_path=tmp_path,
        timezone=audio_timezone,
    )
    timestamps = pd.DatetimeIndex([data.begin for data in audio_dataset.data])
    if aux_timezone is None:
        if timestamps.tz is not None:
            timestamps = timestamps.tz_localize(None)
    else:
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize(aux_timezone)
        else:
            timestamps = timestamps.tz_convert(aux_timezone)

    csv_path = tmp_path / "auxiliary_timezone_mismatch.csv"
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": np.arange(len(timestamps), dtype=float),
        },
    ).to_csv(csv_path, index=False)

    auxiliary_dataset = AuxiliaryDataset.from_audio_dataset(
        audio_dataset=audio_dataset,
        files=[csv_path],
        aggregation="mean",
    )

    observed = [data.to_frame().loc[0, "temperature"] for data in auxiliary_dataset.data]
    expected = list(np.arange(len(audio_dataset.data), dtype=float))

    assert auxiliary_dataset.data[0].to_frame().loc[0, "timestamp"] == audio_dataset.data[0].begin
    assert observed == pytest.approx(expected)


def test_auxiliary_file_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "auxiliary.csv"
    _ = _make_auxiliary_csv(csv_path)

    auxiliary_file = AuxiliaryFile(path=csv_path, time_column="timestamp")

    assert auxiliary_file.begin == Timestamp("2024-01-01 00:00:00")
    assert auxiliary_file.end == Timestamp("2024-01-01 00:00:02")

    subset = auxiliary_file.read(
        start=Timestamp("2024-01-01 00:00:00.300000"),
        stop=Timestamp("2024-01-01 00:00:00.800000"),
    )
    assert list(subset["temperature"]) == [0, 10, 10, 10, 10]


def test_auxiliary_file_from_netcdf(tmp_path: Path) -> None:
    netcdf_path = tmp_path / "auxiliary.nc"
    with netcdf_file(netcdf_path, mode="w") as dataset:
        dataset.createDimension("time", 4)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "seconds since 2024-01-01 00:00:00"
        time[:] = np.arange(4)
        temperature = dataset.createVariable("temperature", "f8", ("time",))
        temperature[:] = np.array([1.0, 2.0, 3.0, 4.0])

    auxiliary_file = AuxiliaryFile(path=netcdf_path, time_column="time")

    assert auxiliary_file.begin == Timestamp("2024-01-01 00:00:00")
    assert auxiliary_file.end == Timestamp("2024-01-01 00:00:04")

    subset = auxiliary_file.read(
        start=Timestamp("2024-01-01 00:00:01"),
        stop=Timestamp("2024-01-01 00:00:03"),
    )
    assert list(subset["temperature"]) == [2.0, 3.0]


def test_auxiliary_file_from_hdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hdf_path = tmp_path / "auxiliary.hdf"
    hdf_path.touch()

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=Timestamp("2024-01-01 00:00:00"),
                periods=4,
                freq="1s",
            ),
            "temperature": [1.0, 2.0, 3.0, 4.0],
        },
    )

    monkeypatch.setattr(pd, "read_hdf", lambda *args, **kwargs: frame)

    auxiliary_file = AuxiliaryFile(
        path=hdf_path,
        time_column="timestamp",
        hdf_key="measurements",
    )

    assert auxiliary_file.begin == Timestamp("2024-01-01 00:00:00")
    assert auxiliary_file.end == Timestamp("2024-01-01 00:00:04")

    subset = auxiliary_file.read(
        start=Timestamp("2024-01-01 00:00:01"),
        stop=Timestamp("2024-01-01 00:00:03"),
    )
    assert list(subset["temperature"]) == [2.0, 3.0]
