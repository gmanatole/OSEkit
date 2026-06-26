"""Surface wind speed estimation from spectrograms and aligned auxiliary data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from osekit.core.auxiliary_dataset import AuxiliaryDataset
    from osekit.core.spectro_data import SpectroData
    from osekit.core.spectro_dataset import SpectroDataset

__all__ = [
    "WIND_METHODS",
    "SurfaceWindEstimator",
    "build_surface_wind_frame",
    "estimate_surface_wind",
    "extract_surface_wind_feature",
    "linear",
    "logarithmic",
    "quadratic",
]


FeatureAggregation = Literal["mean", "median", "max"]
TimestampPosition = Literal["begin", "center", "end"]

_DEFAULT_REFERENCE_FREQUENCY = 8_000.0


def logarithmic(
    x: np.ndarray,
    a: float,
    b: float,
    offset: float,
    *,
    reference_frequency: float = _DEFAULT_REFERENCE_FREQUENCY,
) -> np.ndarray:
    """Logarithmic empirical model used by the wind estimator."""
    if reference_frequency <= 0:
        msg = "The reference frequency must be positive."
        raise ValueError(msg)
    exponent = ((x - offset) + 10 * a * np.log10(reference_frequency)) / (20 * b + 1e-8)
    return 10**exponent


def quadratic(
    x: np.ndarray,
    a: float,
    b: float,
    c: float,
    offset: float,
) -> np.ndarray:
    """Quadratic empirical model used by the wind estimator."""
    return a * (x - offset) ** 2 + b * (x - offset) + c


def linear(
    x: np.ndarray,
    a: float,
    b: float,
    offset: float,
) -> np.ndarray:
    """Linear empirical model used by the wind estimator."""
    return a * (x - offset) + b


WIND_METHODS: dict[str, dict[str, object]] = {
    "Hildebrand": {
        "reference_frequency": _DEFAULT_REFERENCE_FREQUENCY,
        "function": logarithmic,
        "parameters": {"a": 78.0, "b": 1.5},
    },
    "Pensieri": {
        "reference_frequency": _DEFAULT_REFERENCE_FREQUENCY,
        "function": quadratic,
        "parameters": {"a": 0.044642, "b": -3.2917, "c": 63.016},
    },
    "Pensieri_low": {
        "reference_frequency": _DEFAULT_REFERENCE_FREQUENCY,
        "function": linear,
        "parameters": {"a": 0.1458, "b": -3.146},
    },
}


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _top_k_mean(values: np.ndarray, k: int = 50) -> float:
    values = _finite_values(values)
    if values.size == 0:
        return float("nan")
    k = min(k, values.size)
    return float(np.mean(np.sort(values)[-k:]))


def _aggregate_values(values: np.ndarray, aggregation: FeatureAggregation) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    if aggregation == "mean":
        return float(np.nanmean(finite))
    if aggregation == "median":
        return float(np.nanmedian(finite))
    if aggregation == "max":
        return _top_k_mean(finite)
    msg = f"Unsupported aggregation method: {aggregation}"
    raise ValueError(msg)


def _window_timestamp(
    begin: pd.Timestamp,
    end: pd.Timestamp,
    timestamp_position: TimestampPosition,
) -> pd.Timestamp:
    if timestamp_position == "begin":
        return begin
    if timestamp_position == "end":
        return end
    return begin + (end - begin) / 2


def extract_surface_wind_feature(
    spectro_data: SpectroData,
    *,
    frequency: float,
    aggregation: FeatureAggregation = "median",
    band_width_hz: float | None = None,
) -> float:
    """Extract a scalar acoustic feature from one spectrogram."""
    if spectro_data.fft is None:
        msg = "The spectrogram must have an FFT to extract a wind feature."
        raise ValueError(msg)

    values = np.asarray(spectro_data.get_db_value())
    if values.size == 0:
        return float("nan")

    freq = np.asarray(spectro_data.fft.f, dtype=float)
    if freq.size == 0:
        return float("nan")

    if band_width_hz is not None and band_width_hz > 0:
        mask = np.abs(freq - frequency) <= band_width_hz / 2
        if not np.any(mask):
            mask = np.zeros_like(freq, dtype=bool)
            mask[np.argmin(np.abs(freq - frequency))] = True
    else:
        mask = np.zeros_like(freq, dtype=bool)
        mask[np.argmin(np.abs(freq - frequency))] = True

    selected = values[mask, :]
    return _aggregate_values(selected, aggregation)


def _feature_frame_from_spectro_dataset(
    spectro_dataset: SpectroDataset,
    *,
    frequency: float,
    aggregation: FeatureAggregation,
    band_width_hz: float | None,
    timestamp_position: TimestampPosition,
) -> pd.DataFrame:
    rows = [
        {
            "window_begin": spectro_data.begin,
            "window_end": spectro_data.end,
            "spectro_timestamp": _window_timestamp(
                spectro_data.begin,
                spectro_data.end,
                timestamp_position,
            ),
            "acoustic_level": extract_surface_wind_feature(
                spectro_data,
                frequency=frequency,
                aggregation=aggregation,
                band_width_hz=band_width_hz,
            ),
        }
        for spectro_data in spectro_dataset.data
    ]
    return pd.DataFrame(rows)


def _frame_from_auxiliary_dataset(auxiliary_dataset: AuxiliaryDataset) -> pd.DataFrame:
    if not auxiliary_dataset.data:
        return pd.DataFrame()
    frames = [data.to_frame() for data in auxiliary_dataset.data]
    return pd.concat(frames, ignore_index=True, sort=False)


def build_surface_wind_frame(  # noqa: PLR0913
    spectro_dataset: SpectroDataset,
    auxiliary_dataset: AuxiliaryDataset,
    *,
    frequency: float,
    aggregation: FeatureAggregation = "median",
    band_width_hz: float | None = None,
    timestamp_position: TimestampPosition = "begin",
    wind_speed_column: str = "wind_speed",
) -> pd.DataFrame:
    """Build a training frame from aligned spectrograms and auxiliary target values."""
    spectro_frame = _feature_frame_from_spectro_dataset(
        spectro_dataset,
        frequency=frequency,
        aggregation=aggregation,
        band_width_hz=band_width_hz,
        timestamp_position=timestamp_position,
    )
    auxiliary_frame = _frame_from_auxiliary_dataset(auxiliary_dataset)

    if spectro_frame.empty or auxiliary_frame.empty:
        msg = "The spectrogram and auxiliary datasets must not be empty."
        raise ValueError(msg)

    if wind_speed_column not in auxiliary_frame.columns:
        msg = (
            "The auxiliary dataset must contain a "
            f"'{wind_speed_column}' column."
        )
        raise ValueError(msg)

    merged = spectro_frame.merge(
        auxiliary_frame,
        on=["window_begin", "window_end"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        msg = "No aligned spectrogram / auxiliary rows could be matched."
        raise ValueError(msg)
    return merged.sort_values("window_begin").reset_index(drop=True)


def _parameter_bounds(
    initial_values: Sequence[float],
    *,
    scaling_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for value in initial_values:
        span = abs(float(value)) * scaling_factor
        if span == 0:
            span = max(scaling_factor, 1e-6)
        lower.append(float(value) - span)
        upper.append(float(value) + span)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total == 0:
        return float("nan")
    return float(1 - residual / total)


@dataclass
class SurfaceWindEstimator:
    """Fit an empirical mapping from spectrogram level to surface wind speed."""

    method: str = "Pensieri_low"
    frequency: float | None = None
    aggregation: FeatureAggregation = "median"
    band_width_hz: float | None = None
    timestamp_position: TimestampPosition = "begin"
    feature_offset: float | None = None
    scaling_factor: float = 0.2
    maxfev: int = 25_000
    wind_speed_column: str = "wind_speed"

    def __post_init__(self) -> None:
        """Validate the configuration and initialize the fit state."""
        if self.method not in WIND_METHODS:
            msg = f"Unsupported wind method: {self.method}"
            raise ValueError(msg)
        if self.aggregation not in {"mean", "median", "max"}:
            msg = f"Unsupported aggregation method: {self.aggregation}"
            raise ValueError(msg)
        if self.timestamp_position not in {"begin", "center", "end"}:
            msg = f"Unsupported timestamp position: {self.timestamp_position}"
            raise ValueError(msg)
        method_spec = WIND_METHODS[self.method]
        self.reference_frequency = float(method_spec["reference_frequency"])
        self.frequency = (
            self.reference_frequency
            if self.frequency is None
            else float(self.frequency)
        )
        if not self.wind_speed_column:
            msg = "The wind speed column name must be non-empty."
            raise ValueError(msg)
        self._initial_parameters = tuple(
            float(value) for value in method_spec["parameters"].values()
        )
        self.params_: dict[str, float] | None = None
        self.offset_: float | None = self.feature_offset
        self.stats_: dict[str, float] = {}
        self.training_frame_: pd.DataFrame | None = None
        self.fitted_: bool = False

    def _curve_function(self, x: np.ndarray, *params: float) -> np.ndarray:
        if self.method == "Hildebrand":
            a, b = params
            return logarithmic(
                x,
                a,
                b,
                offset=float(self.offset_),
                reference_frequency=self.reference_frequency,
            )
        if self.method == "Pensieri":
            a, b, c = params
            return quadratic(x, a, b, c, offset=float(self.offset_))
        a, b = params
        return linear(x, a, b, offset=float(self.offset_))

    def _finalize_output_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if "spectro_timestamp" in frame.columns:
            if "timestamp" in frame.columns:
                frame = frame.drop(columns=["spectro_timestamp"])
            else:
                frame = frame.rename(columns={"spectro_timestamp": "timestamp"})
        frame["wind_model"] = self.method
        frame["acoustic_frequency_hz"] = self.frequency
        frame["feature_aggregation"] = self.aggregation
        frame["feature_offset"] = self.offset_
        if self.band_width_hz is not None:
            frame["feature_bandwidth_hz"] = self.band_width_hz
        if self.wind_speed_column in frame.columns:
            frame[f"{self.wind_speed_column}_residual"] = (
                frame[self.wind_speed_column] - frame["wind_speed_estimated"]
            )
        return frame

    def build_training_frame(
        self,
        spectro_dataset: SpectroDataset,
        auxiliary_dataset: AuxiliaryDataset,
    ) -> pd.DataFrame:
        """Build the merged frame used to fit the wind estimator."""
        return build_surface_wind_frame(
            spectro_dataset=spectro_dataset,
            auxiliary_dataset=auxiliary_dataset,
            frequency=self.frequency,
            aggregation=self.aggregation,
            band_width_hz=self.band_width_hz,
            timestamp_position=self.timestamp_position,
            wind_speed_column=self.wind_speed_column,
        )

    def fit(
        self,
        spectro_dataset: SpectroDataset,
        auxiliary_dataset: AuxiliaryDataset,
    ) -> Self:
        """Fit the empirical wind model."""
        frame = self.build_training_frame(
            spectro_dataset=spectro_dataset,
            auxiliary_dataset=auxiliary_dataset,
        )
        frame = frame.dropna(
            subset=["acoustic_level", self.wind_speed_column],
        )
        frame = frame.reset_index(drop=True)
        if frame.empty:
            msg = "No usable rows are available to fit the wind estimator."
            raise ValueError(msg)

        x = frame["acoustic_level"].to_numpy(dtype=float)
        y = frame[self.wind_speed_column].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        frame = frame.loc[finite].reset_index(drop=True)
        x = x[finite]
        y = y[finite]
        if x.size < len(self._initial_parameters):
            msg = "Not enough samples to fit the selected wind model."
            raise ValueError(msg)

        if self.offset_ is None:
            self.offset_ = float(np.min(x))

        p0 = np.asarray(self._initial_parameters, dtype=float)
        bounds = _parameter_bounds(p0, scaling_factor=self.scaling_factor)
        popt, _ = curve_fit(
            self._curve_function,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=self.maxfev,
        )
        predicted = self._curve_function(x, *popt)
        self.params_ = {
            name: float(value)
            for name, value in zip(self._parameter_names(), popt, strict=False)
        }
        self.training_frame_ = self._finalize_output_frame(
            frame.assign(wind_speed_estimated=predicted),
        )
        self.stats_ = {
            "n_samples": len(frame),
            "mae": _mean_absolute_error(y, predicted),
            "rmse": _root_mean_squared_error(y, predicted),
            "r2": _r2_score(y, predicted),
        }
        self.fitted_ = True
        return self

    def _parameter_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in WIND_METHODS[self.method]["parameters"])

    def predict(
        self,
        spectro_dataset: SpectroDataset,
        *,
        auxiliary_dataset: AuxiliaryDataset | None = None,
    ) -> pd.DataFrame:
        """Predict surface wind speed for a spectrogram dataset."""
        if not self.fitted_ or self.params_ is None or self.offset_ is None:
            msg = "The estimator must be fitted before calling predict()."
            raise RuntimeError(msg)

        frame = _feature_frame_from_spectro_dataset(
            spectro_dataset,
            frequency=self.frequency,
            aggregation=self.aggregation,
            band_width_hz=self.band_width_hz,
            timestamp_position=self.timestamp_position,
        )
        frame["wind_speed_estimated"] = self._curve_function(
            frame["acoustic_level"].to_numpy(dtype=float),
            *[self.params_[name] for name in self._parameter_names()],
        )

        if auxiliary_dataset is not None:
            aux_frame = _frame_from_auxiliary_dataset(auxiliary_dataset)
            frame = frame.merge(
                aux_frame,
                on=["window_begin", "window_end"],
                how="inner",
                validate="one_to_one",
            )

        final_frame = self._finalize_output_frame(frame).sort_values("window_begin")
        return final_frame.reset_index(drop=True)

    def fit_predict(
        self,
        spectro_dataset: SpectroDataset,
        auxiliary_dataset: AuxiliaryDataset,
    ) -> pd.DataFrame:
        """Fit the estimator and return predictions on the training dataset."""
        self.fit(spectro_dataset=spectro_dataset, auxiliary_dataset=auxiliary_dataset)
        return self.predict(
            spectro_dataset=spectro_dataset,
            auxiliary_dataset=auxiliary_dataset,
        )

    def write(
        self,
        frame: pd.DataFrame,
        folder: Path,
        *,
        filename: str | None = None,
    ) -> Path:
        """Write a predicted wind frame to CSV."""
        folder.mkdir(parents=True, exist_ok=True)
        filename = "surface_wind_speed.csv" if filename is None else filename
        output = folder / filename
        if output.suffix.lower() != ".csv":
            output = output.with_suffix(".csv")
        frame.to_csv(output, index=False)
        return output

    def export(
        self,
        spectro_dataset: SpectroDataset,
        auxiliary_dataset: AuxiliaryDataset,
        *,
        folder: Path | None = None,
        filename: str | None = None,
    ) -> tuple[pd.DataFrame, Path | None]:
        """Fit, predict and optionally write the results to disk."""
        frame = self.fit_predict(
            spectro_dataset=spectro_dataset,
            auxiliary_dataset=auxiliary_dataset,
        )
        output_path: Path | None = None
        if folder is not None:
            output_path = self.write(frame, folder=folder, filename=filename)
        return frame, output_path


def estimate_surface_wind(  # noqa: PLR0913
    spectro_dataset: SpectroDataset,
    auxiliary_dataset: AuxiliaryDataset,
    *,
    method: str = "Pensieri_low",
    frequency: float | None = None,
    aggregation: FeatureAggregation = "median",
    band_width_hz: float | None = None,
    timestamp_position: TimestampPosition = "begin",
    wind_speed_column: str = "wind_speed",
    feature_offset: float | None = None,
    scaling_factor: float = 0.2,
    maxfev: int = 25_000,
    folder: Path | None = None,
    filename: str | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Estimate surface wind speed and optionally write a CSV."""
    estimator = SurfaceWindEstimator(
        method=method,
        frequency=frequency,
        aggregation=aggregation,
        band_width_hz=band_width_hz,
        timestamp_position=timestamp_position,
        wind_speed_column=wind_speed_column,
        feature_offset=feature_offset,
        scaling_factor=scaling_factor,
        maxfev=maxfev,
    )
    return estimator.export(
        spectro_dataset=spectro_dataset,
        auxiliary_dataset=auxiliary_dataset,
        folder=folder,
        filename=filename,
    )
