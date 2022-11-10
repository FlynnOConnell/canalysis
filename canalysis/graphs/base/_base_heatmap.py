from __future__ import annotations

import matplotlib.pyplot as plt
from ..base import _base_figure 
from ..graph_utils import helpers

helpers.update_rcparams()


class BaseHeatmap:
    def __init__(
        self,
        cmap: str = "plasma",
        colorbar: bool = False,
        fig: plt.figure = None,
        ax: plt.axes = None,
        **figargs
    ):
        self.cmap = plt.get_cmap(cmap)
        self.colorbar = colorbar
        self.fig = plt.figure(FigureClass=_base_figure.CalFigure) if fig is None else fig
        self.ax = self.fig.add_subplot(111) if ax is None else ax

    def clear(self):
        self.ax.clear()
