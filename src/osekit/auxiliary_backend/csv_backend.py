import warnings
from os import PathLike

import numpy as np
import pandas as pd


class CSVBackend:
    """Backend for reading CSV files."""

    def __init__(self) -> None:
        """Initialize the CSV backend."""

    @property
    def variables(self) -> list:
        return self._variables

    @variables.setter
    def variables(self, var = list[str] | str) -> None:
        if var not in self.columns :
            raise ValueError(f"Variable {var} is not in the CSV file.")
        self._variables = [var] if isinstance(var, str) else var

    def close(self) -> None:
        """Close the currently opened file."""


    def info(self, path: PathLike | str, timestamp_col : str) -> tuple[int, int, int, int]:
        """Return the sample rate, number of frames and channels of the CSV file.

        Parameters
        ----------
        path: PathLike | str
            Path to the auxiliary file.

        Returns
        -------
        tuple[int,int,int]:
            Sample rate, number of frames, number of variables and duration of the CSV file.

        """
        file_content = pd.read_csv(path, parse_dates = [timestamp_col])

        self.columns = list(file_content.columns)

        sample_rate = file_content[timestamp_col].diff().dt.total_seconds().to_numpy()[1:]
        if len(np.unique(sample_rate)) != 1:
            msg = "Inconsistent sampling rates in CSV file."
            warnings.warn(msg)
            sample_rate = np.nan
        else :
            sample_rate = int(np.unique(sample_rate).mean())
        duration = (file_content[timestamp_col].iloc[-1] - file_content[timestamp_col].iloc[0]).total_seconds()
        frames = len(file_content)
        return (
            sample_rate,
            frames,
            len(self.columns) - 1,
            int(duration),
        )

    def read_timestamps(self, path: PathLike | str, timestamp_col : str) -> pd.Series :
        """Return the timestamp column of auxiliary file.

        Parameters
        ----------
        path: PathLike | str
            Path to the auxiliary file.
        timestamp_col: str
            Name of the timestamp column.
        Returns
        -------
        pd.Series:
            pd.Series containing the timestamp column.
        """
        file_content = pd.read_csv(path, parse_dates = [timestamp_col])
        return file_content[timestamp_col]

    def read(
        self,
        path: PathLike | str,
        start: int = 0,
        stop: int | None = None,
    ) -> np.ndarray:
        """Read the content of a CSV file.

        Parameters
        ----------
        path: PathLike | str
            Path to the audio file.
        start: int
            First frame to read.
        stop: int
            Frame after the last frame to read.

        Returns
        -------
        np.ndarray:
            A ``(channel * frames)`` array containing the CSV data.

        """
        file_content = pd.read_csv(path)
        data = file_content[self.variables].to_numpy()

        self.columns = list(file_content.columns)

        return data[start:stop]
