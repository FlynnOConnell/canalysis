#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# heatmaps.py
"""

from __future__ import annotations

from graph_utils import helpers
from base._base_heatmap import BaseHeatmap

import numpy as np
import seaborn as sns
from utils import funcs

helpers.update_rcparams()


class EatingHeatmap(BaseHeatmap):
    def __init__(
        self,
        data,
        premask: np.ndarray = None,
        cmap: str = "plasma",
        colorbar: bool = False,
        save_dir: str | None = "",
        title: str | None = "",
        **figargs
    ):
        super().__init__(cmap, colorbar, **figargs)
        self.data = data
        self.data.columns = np.round(np.arange(0, len(self.data.columns) / 10, 0.1), 1)
        self.premask = premask
        self.title = title
        self.save_dir = save_dir
        """
        Create heatmaps.

        .. note::
            -Using seaborn(sns).heatmap(), each row of the heatmap needs to correspond to each row of dictionary.
            -If incoming data_dict is already in this format, you can just delete the df = df.T.
            -Using matplotlib.pyplot(plt).imshow(), each column of the heatmap corresponds to columns of the data.
        Parameters
        ----------
        savedir: str = ''
            Path to save the file.
        cm: str = 'inferno'
            Colormap color. Options:
             1) https://seaborn.pydata.org/generated/seaborn.color_palette.html?highlight=sns%20color_palette
             2) https://matplotlib.org/stable/tutorials/colors/colormaps.html
        _id : str
            Identifier that will be appended to the filenames this batch of graphs.
                -e.g. _id = 'heatmaps' -> .../heatmaps.png
        title: str = ''
            If parameter passed, set this string as the plot title.
        sigma : scalar or sequence of scalars = 0
            Standard deviation for Gaussian kernal. This smooths out the heatmap to give a more
            uniform distribution. Set to 0 for no smoothing.
        square : bool = False
            If True, set the Axes' aspect to “equal” so each cell will be square-shaped.
        colorbar : bool = False
            If true, a colorbar will be plotted alongside each heatmap.
        robust: bool = False
            If true, and cbar = True, the colormap range is computed with robust quantiles instead of the extreme values
        line_loc : Optional[int] = 0
            If an int is set, plot a vertical(or horizontal) line at the x(or y) value indicated by the given number.
        linewidth : Optional[int] = 3
            Width of vertical line if line_loc is set.
        line_color: str = 'white'
            Color of vertical line, if line_loc is set.
        """

    @property
    def ax(self, ):
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax = value

    def show(self):
        self.fig.show()

    def save(self) -> None:
        file = f"{self.save_dir}.png"
        savefile = funcs.check_unique_path(file)
        self.fig.savefig(f"{savefile}", dpi=400, bbox_inches="tight", pad_inches=0.01, )
        return None

    def default_heatmap(self,
                        eatingstart,
                        entrystart,
                        eatingend,
                        **kwargs
                        ):
        """Plot single heatmap with seaborn library."""

        self.ax.set_title(self.title, fontweight="bold")
        self.data = self.data[self.data.columns].astype(float)
        self.ax = sns.heatmap(self.data, cbar=self.colorbar, cmap=self.cmap, mask=self.data.isnull(), **kwargs)
        self.set_heatmap_lines(eatingstart, entrystart, eatingend)
        self.ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        if self.premask is not None:
            self.ax.set_xticks(self.premask)
            self.ax.xaxis.set_ticklabels(ticklabels=self.premask, rotation=45)
        else:
            self.ax.xaxis.set_ticklabels(ticklabels=self.ax.get_xticklabels(), rotation=45)
        self.ax.set_yticks(list(i + 0.5 for i in range(0, self.data.shape[0])))
        self.ax.set_yticklabels(list(self.data.index.values))
        n = 2
        [l.set_visible(False) for (i, l) in enumerate(self.ax.xaxis.get_ticklabels()) if i % n != 0]
        if self.save_dir:
            self.save()
        # plt.close(self.fig)
        self.fig.close()
        return self.fig

    def set_heatmap_lines(self, eatingstart, entrystart, eatingend):
        line_loc1 = (eatingstart - entrystart) * 10
        line_loc2 = (eatingend - eatingstart) * 10
        self.ax.axvline(
                line_loc1,
                color='k',
                linewidth=2,)
        self.ax.axvline(
                line_loc2,
                color='k',
                linewidth=2,)
        return None
