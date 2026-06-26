"""Auxiliary item corresponding to a portion of an ``AuxiliaryFile``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from osekit.core.auxiliary_file import AuxiliaryFile
from osekit.core.base_item import BaseItem


class AuxiliaryItem(BaseItem[AuxiliaryFile]):
    """Auxiliary item corresponding to a portion of an ``AuxiliaryFile``."""

    def get_value(self) -> pd.DataFrame:
        """Return the auxiliary rows included in this item."""
        if self.is_empty:
            return pd.DataFrame()
        return self.file.read(start=self.begin, stop=self.end)
