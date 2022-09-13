from __future__ import annotations

import matplotlib.pyplot as plt
from base._base_figure import CalFigure
from graph_utils import helpers
from utils import funcs

helpers.update_rcparams()


class BaseHeatmap:
    def __init__(
        self,
        save_dir: str = '',
        save_id: str = '',
        cmap: str = "plasma",
        sigma: int | None = None,
        colorbar: bool = False,
        fig: plt.figure = None,
        ax: plt.axes = None
    ):
        self.save_dir = save_dir
        self.save_id = save_id
        self.cmap = plt.get_cmap(cmap)
        self.sigma = sigma
        self.colorbar = colorbar
        self.fig = plt.figure(FigureClass=CalFigure) if fig is None else fig
        self.ax = self.fig.add_subplot(111) if ax is None else ax

    def show_heatmap(self):
        self.fig.show()

    def save(self):
        file = f"{self.save_dir}/{self.save_id}.png"
        savefile = funcs.check_unique_path(file)
        plt.savefig(f"{savefile}", dpi=400, bbox_inches="tight", pad_inches=0.01, )

    def clear(self):
        self.ax.clear()
