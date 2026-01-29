"""Audio File Manager which keeps an audio file open until a request in another file is made.

This workflow avoids closing/opening a same file repeatedly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from osekit.auxiliary_backend.csv_backend import CSVBackend
from osekit.auxiliary_backend.netcdf_backend import NetCDFBackend

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np


class AuxiliaryFileManager:
    """Auxiliary File Manager which keeps an auxiliary file open until a request in another file is made."""

    def __init__(self) -> None:
        """Initialize an auxiliary file manager."""
        self._csv = CSVBackend()
        self._netcdf : NetCDFBackend | None = None

    def close(self) -> None:
        """Close the currently opened file."""
        self._csv.close()
        if self._netcdf:
            self._netcdf.close()

    def _backend(self, path: PathLike | str) -> CSVBackend | NetCDFBackend:
        suffix = Path(path).suffix.lower()

        if suffix == ".nc":
            if self._netcdf is None:
                self._netcdf = NetCDFBackend()
            return self._netcdf

        return self._csv

    def info(self, path: PathLike | str, timestamp_col : str) -> tuple[int, int, int]:
        """Return the sample rate, number of frames and number of variables of the auxiliary file.

        Parameters
        ----------
        path: PathLike | str
            Path to the audio file.
        timestamp_col: str
            Name of the timestamp column.

        Returns
        -------
        tuple[int,int,int]:
            Sample rate, number of frames and number of variables of the auxiliary file.

        """
        return self._backend(path).info(path, timestamp_col)

    def read(
        self,
        path: PathLike | str,
        start: int = 0,
        stop: int | None = None,
    ) -> np.ndarray:
        """Read the content of an auxiliary file.

        If the auxiliary file is not the current opened file,
        the current opened file is switched.

        Parameters
        ----------
        path: PathLike | str
            Path to the auxiliary file.
        start: int
            First frame to read.
        stop: int | None
            Frame after the last frame to read.

        Returns
        -------
        np.ndarray:
            A ``(number of variables * frames)`` array containing the auxiliary data.

        """
        _, frames, _ = self.info(path)

        if stop is None:
            stop = frames

        if stop is None:
            stop = frames

        if not 0 <= start < frames:
            msg = "Start should be between 0 and the last frame of the auxiliary file."
            raise ValueError(msg)
        if not 0 <= stop <= frames:
            msg = "Stop should be between 0 and the last frame of the auxiliary file."
            raise ValueError(msg)
        if start > stop:
            msg = "Start should be inferior to Stop."
            raise ValueError(msg)

        return self._backend(path).read(path=path, start=start, stop=stop)
