import warnings
from os import PathLike

import numpy as np
import pandas as pd


class NetCDFBackend:
    """Backend for reading NetCDF files."""

    def __init__(self) -> None:
        """Initialize the NetCDF backend."""

    @property
    def variables(self) -> list:
        return list(None)

    @variables.setter
    def variables(self, var = list[str] | str) -> None:
        if var not in self.columns :
            raise ValueError(f"Variable {var} is not in the CSV file.")
        self._variables = list(var)

    def close(self) -> None:
        """Close the currently opened file."""


    def info(self, path: PathLike | str, timestamp_col : str = "timestamps") -> tuple[int, int, int]:
        """Return the sample rate, number of frames and channels of the CSV file.

        Parameters
        ----------
        path: PathLike | str
            Path to the auxiliary file.

        Returns
        -------
        tuple[int,int,int]:
            Sample rate, number of frames and channels of the MSEED file.

        """
        file_content = pd.read_csv(path)

        self.columns = list(file_content.columns)

        sample_rate = file_content[timestamp_col].diff().dt.total_seconds().to_numpy()[1:]
        if len(np.unique(sample_rate)) != 1:
            msg = "Inconsistent sampling rates in CSV file."
            warnings.warn(msg)

        frames = len(file_content)
        return (
            int(np.unique(sample_rate).mean()),
            frames,
            len(self.columns) - 1,
        )

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
            A ``(channel * frames)`` array containing the MSEED data.

        """
        file_content = pd.read_csv(path)
        data = file_content[self.var_name].to_numpy()

        self.columns = list(file_content.columns)

        return data[start:stop]
