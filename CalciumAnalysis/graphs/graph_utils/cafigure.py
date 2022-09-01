
from __future__ import annotations

import matplotlib


class CalFigure(matplotlib.figure.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.set_dpi(dpi)
        self.tight_layout()
        self.frameon = False

