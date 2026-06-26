from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from osekit.config import TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
from osekit.core.audio_dataset import AudioDataset
from osekit.core.auxiliary_dataset import AuxiliaryDataset
from osekit.core.spectro_dataset import SpectroDataset
from osekit.public.wind import (
    SurfaceWindEstimator,
    estimate_surface_wind,
    extract_surface_wind_feature,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "audio_files",
    [
        {
            "duration": 1,
            "sample_rate": 1_024,
            "nb_files": 4,
            "date_begin": pd.Timestamp("2024-01-01 00:00:00"),
            "series_type": "increase",
        },
    ],
    indirect=True,
)
def test_surface_wind_estimator_fit_predict_and_export(
    tmp_path: Path,
    audio_files: tuple[list, pytest.fixtures.Subrequest],
) -> None:
    audio_dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
        mode="files",
    )
    sft = ShortTimeFFT(hamming(128), hop=64, fs=1_024)
    spectro_dataset = SpectroDataset.from_audio_dataset(audio_dataset, fft=sft)

    target_frequency = float(spectro_dataset.fft.f[1])

    rows = []
    for spectro_data in spectro_dataset.data:
        feature = extract_surface_wind_feature(
            spectro_data,
            frequency=target_frequency,
            aggregation="median",
        )
        rows.append(
            {
                "timestamp": spectro_data.begin,
                "wind_speed_insitu": 0.1458 * feature - 3.146,
            },
        )

    auxiliary_csv = tmp_path / "auxiliary.csv"
    pd.DataFrame(rows).to_csv(auxiliary_csv, index=False)
    auxiliary_dataset = AuxiliaryDataset.from_audio_dataset(
        audio_dataset=audio_dataset,
        files=[auxiliary_csv],
    )

    estimator = SurfaceWindEstimator(
        method="Pensieri_low",
        frequency=target_frequency,
        aggregation="median",
        wind_speed_column="wind_speed_insitu",
        feature_offset=0.0,
    )
    predicted = estimator.fit_predict(
        spectro_dataset=spectro_dataset,
        auxiliary_dataset=auxiliary_dataset,
    )

    assert estimator.fitted_ is True
    assert estimator.training_frame_ is not None
    assert set(estimator.params_) == {"a", "b"}
    assert {"timestamp", "window_begin", "window_end"} <= set(predicted.columns)
    assert "wind_speed_estimated" in predicted.columns
    assert "wind_speed_insitu" in predicted.columns
    assert "wind_speed_insitu_residual" in predicted.columns
    assert np.allclose(
        predicted["wind_speed_insitu"],
        predicted["wind_speed_estimated"],
    )
    rmse_threshold = 1e-8
    assert estimator.stats_["rmse"] < rmse_threshold

    output_folder = tmp_path / "auxiliary" / "wind"
    output_path = estimator.write(predicted, folder=output_folder)
    assert output_path.exists()
    written = pd.read_csv(output_path)
    assert "wind_speed_estimated" in written.columns
    assert len(written) == len(predicted)

    frame, maybe_path = estimate_surface_wind(
        spectro_dataset,
        auxiliary_dataset,
        method="Pensieri_low",
        frequency=target_frequency,
        aggregation="median",
        wind_speed_column="wind_speed_insitu",
        feature_offset=0.0,
        folder=tmp_path / "auxiliary" / "wind_helper",
    )
    assert maybe_path is not None
    assert maybe_path.exists()
    assert np.allclose(frame["wind_speed_insitu"], frame["wind_speed_estimated"])
