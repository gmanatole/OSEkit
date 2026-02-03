"""Auxiliary file associated with timestamps."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

from math import floor

import numpy as np
import pandas as pd
import pytz
from pandas import Timedelta, Timestamp

from osekit.core_api import auxiliary_file_manager as afm
from osekit.core_api.base_file import BaseFile


class AuxiliaryFile(BaseFile):
    """Auxiliary file associated with timestamps."""

    supported_extensions: typing.ClassVar = [".nc", ".csv"]

    def __init__(
        self,
        path: PathLike | str,
        timestamp_col : str | None = None,
        begin: Timestamp | None = None,
        strptime_format: str | list[str] | None = None,
        timezone: str | pytz.timezone | None = None,
    ) -> None:
        """Initialize an ``AuxiliaryFile`` object with a path and a begin timestamp.

        The begin timestamp can either be provided as a parameter
         or parsed from the filename according to the provided ``strptime_format``.

        Parameters
        ----------
        path: PathLike | str
            Full path to the file.
        begin: pandas.Timestamp | None
            Timestamp corresponding to the first data point in the file.
            If it is not provided, ``strptime_format`` is mandatory.
            If both ``begin`` and ``strptime_format`` are provided,
            ``begin`` will overrule the timestamp embedded in the filename.
        strptime_format: str | None
            The strptime format used in the text.
            It should use valid strftime codes (https://strftime.org/).
            Example: ``'%y%m%d_%H:%M:%S'``.
        timezone: str | pytz.timezone | None
            The timezone in which the file should be localized.
            If ``None``, the file begin/end will be tz-naive.
            If different from a timezone parsed from the filename, the timestamps'
            timezone will be converted from the parsed timezone
            to the specified timezone.

        """
        if not begin and timestamp_col is not None:
            begin = pd.read_csv(path, parse_dates=[timestamp_col]).iloc[0][timestamp_col]
        else :
            raise ValueError("Either specify begin timestamp or timestamp column")
        super().__init__(
            path=path,
            begin=begin,
            strptime_format=strptime_format,
            timezone=timezone,
        )
        self.timestamp_col = timestamp_col
        sample_rate, frames, nvars, duration = afm.info(path, timestamp_col)
        self.sample_rate = sample_rate
        self.nvars = nvars
        self.end = self.begin + Timedelta(seconds=duration)

    def read(self, start: Timestamp, stop: Timestamp) -> np.ndarray:
        """Return the auxiliary data between start and stop from the file.

        Parameters
        ----------
        start: pandas.Timestamp
            Timestamp corresponding to the first data point to read.
        stop: pandas.Timestamp
            Timestamp after the last data point to read.

        Returns
        -------
        numpy.ndarray:
            The audio data between ``start`` and ``stop``.
            The first frame of the data is the first frame that ends after ``start``.
            The last frame of the data is the last frame that starts before ``stop``.

        """
        start_sample, stop_sample = self.frames_indexes(start, stop)
        data = afm.read(self.path, start=start_sample, stop=stop_sample)
        if len(data.shape) == 1:
            return data.reshape(
                data.shape[0],
                1,
            )  # 2D array to match the format of multichannel audio
        return data

    def frames_indexes(self, start: Timestamp, stop: Timestamp) -> tuple[int, int]:
        """Return the indexes of the frames between the ``start`` and ``stop`` timestamps.

        The ``start`` index is that of the first sample that ends after the ``start``
        timestamp.
        The ``stop`` index is that of the last sample that starts before the ``stop``
        timestamp.

        Parameters
        ----------
        start: pandas.Timestamp
            Timestamp corresponding to the first data point to read.
        stop: pandas.Timestamp
            Timestamp after the last data point to read.

        Returns
        -------
        tuple[int,int]
            First and last frames of the data.

        """
        if np.isnan(self.sample_rate):
            timestamps = afm.read_timestamps(self.path, timestamp_col=self.timestamp_col)
            start_sample = timestamps.searchsorted(start) - 1
            stop_sample = timestamps.searchsorted(stop)
        else :
            start_sample = floor(((start - self.begin) * self.sample_rate).total_seconds())
            stop_sample = round(((stop - self.begin) * self.sample_rate).total_seconds())
        return start_sample, stop_sample

    def move(self, folder: Path) -> None:
        """Move the file to the target folder.

        Parameters
        ----------
        folder: Path
            destination folder where the file will be moved.

        """
        afm.close()
        super().move(folder)
