#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# heatmaps.py
"""

from __future__ import annotations

from typing import Optional, ClassVar
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from scipy.ndimage.filters import gaussian_filter
from utils import funcs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def set_pub():
    """Update matplotlib backend default styles to be bigger and bolder."""
    rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.facecolor": "w",
            "axes.labelsize": 15,
            "lines.linewidth": 1,
        }
    )


# %%

class BaseHeatmap:
    def __init__(
            self,
            data,
            save_dir: str = '',
            save_id: str = '',
            cmap: str = "RdBu",
            sigma: int | None = None,
            colorbar: bool = False,
    ):
        self.save_dir = save_dir
        self.save_id = save_id
        self.cmap = plt.get_cmap(cmap)
        self.sigma = sigma
        self.colorbar = colorbar

    @staticmethod
    def show_heatmap():
        plt.show()

    def save(self):
        file = f"{self.save_dir}/{self.save_id}.png"
        savefile = funcs.check_unique_path(file)
        plt.savefig(f"{savefile}", dpi=400, bbox_inches="tight", pad_inches=0.01, )


class EatingHeatmap(BaseHeatmap):
    def __init__(
            self,
            data,
            save_dir: str | None = "",
            title: str | None = "",
    ):
        super().__init__(save_dir)
        self.data = data
        self.fig, self.ax = plt.subplots()
        self.title = title

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
    def ax(self,):
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax = value

    def interval_heatmap(self, eatingstart, entrystart, eatingend, **kwargs):
        """
        Plot single heatmap with seaborn library.
        """
        set_pub()
        if self.sigma:
            self.data = pd.DataFrame(gaussian_filter(self.data, sigma=self.sigma))
        self.ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        self.ax.set_title('Z scores', fontweight="bold")
        self.set_heatmap_lines(eatingstart, entrystart, eatingend)
        sns.heatmap(self.data, cbar=self.colorbar, cmap=self.cmap, **kwargs)

    def set_heatmap_lines(self, eatingstart, entrystart, eatingend):
        my_ticks = self.ax.get_xticks()
        line_loc1 = (eatingstart - entrystart) * 10,
        line_loc2 = (eatingend - eatingstart) * 10,
        self.ax.set_xticks([my_ticks[0],
                            my_ticks[-1]],
                           ['{:.2f}'.format(0),
                            '{:.2f}'.format(my_ticks[-1] / 10)],
                           visible=True,
                           rotation="horizontal")
        self.ax.axvline(
            line_loc1,
            color='k',
            linewidth=3.5,
            alpha=0.9)
        self.ax.axvline(
            line_loc2,
            color='k',
            linewidth=3.5,
            alpha=0.9)
        return None
