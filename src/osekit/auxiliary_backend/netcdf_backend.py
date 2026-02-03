import warnings
from os import PathLike

import numpy as np
import netCDF4 as nc
### NETCDF or xarray ???


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



    def info(self, path: PathLike | str, timestamp_col : str = "timestamps") -> tuple[int, int, int, int]:
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
        file_content = nc.Dataset(path)

        self.columns = list(file_content.variables.keys())

        sample_rate = file_content[timestamp_col][:].filled(np.nan).diff().dt.total_seconds().to_numpy()[1:]
        if len(np.unique(sample_rate)) != 1:
            msg = "Inconsistent sampling rates in CSV file."
            warnings.warn(msg)
        else :
            sample_rate = int(np.unique(sample_rate).mean())
        duration = (file_content[timestamp_col].iloc[-1] - file_content[timestamp_col].iloc[0]).total_seconds()
        frames = len(file_content)
        return (
            int(np.unique(sample_rate).mean()),
            frames,
            len(self.columns) - 1,
            int(duration)
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
        file_content = nc.Dataset(path)
        data = file_content[self.var_name][:].filled(np.nan)

        self.columns = list(file_content.variables.keys())

        return data[start:stop]
