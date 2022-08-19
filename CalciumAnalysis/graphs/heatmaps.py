#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# heatmaps.py
"""

from __future__ import annotations

from typing import Optional
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import LogNorm
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


class Heatmap(object):
    # Initialize the attributes to apply to our heatmaps
    # nearly all optional and can be ignored.
    def __init__(
        self,
        save_dir: str | None = "",
        cm: str = "magma",
        _id: str | None = "",
        title: str | None = "",
        xlabel: str | None = "",
        sigma: int | None = None,
        square: bool = False,
        colorbar: bool = False,
        robust: bool = False,
        line_loc: Optional[int] = 0,
        line_width: Optional[int] = 3,
        line_color: str = "white",
    ):
        self.save_dir = save_dir
        self.cm = plt.get_cmap(cm)
        self._id = _id
        self.title = title
        self.xlabel = xlabel
        self.sigma = sigma
        self.square = square
        self.colorbar = colorbar
        self.robust = robust
        # Attributes for placing a line on the graph
        self.line_loc = line_loc
        self.line_width = line_width
        self.line_color = line_color
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
            
        .. ::usage: 
            my_heatmap = HeatMap('/users/pictures', 'inferno', 'this_cell', 'heatmap')
            my_heatmap.single(data)
        """

    def nested(self, data_dict: dict, **axargs):
        """
        Plot multiple heatmaps from a nested dictionary. One heatmap for each dict.key, with the heatmap title
        corresponding to that key. Within each key is a pandas DataFrame containing the heatmap data.

        .. note::
            Using seaborn (sns), each row of the heatmap needs to correspond to each row of dictionary.
            If incoming data_dict is already in this format, you can just delete the df = df.T.

        Parameters
        ----------
        data_dict : dict
            Data used in heatmap. The structure of this object should be:
            data_dict = {
            graph1 : pd.DataFrame(data),
            graph2 : pd.DataFrame(data)
            } ...etc.
        """
        set_pub()

        for cell, df in data_dict.items():
            df = df.T
            if self.sigma is not None:
                df = pd.DataFrame(gaussian_filter(df, sigma=self.sigma))
            # Plot data
            fig, axs = plt.subplots()
            vmin = min(df[cell])
            vmax = max(df[cell])
            sns.heatmap(
                df,
                square=self.square,
                cbar=self.colorbar,
                robust=self.robust,
                vmin=vmin,
                vmax=vmax,
                **axargs,
            )
            axs.axis("off")
            if self.line_loc:
                axs.axvline(
                    x=self.line_loc, color=self.line_color, linewidth=self.line_width
                )
            if self.save_dir:
                plt.savefig(
                    f"{self.save_dir}/{self._id}.png",
                    dpi=400,
                    bbox_inches="tight",
                    pad_inches=0.01,
                )
            return fig

    def columnwise(self, df: pd.DataFrame, axs: object = None):
        """
        Plot heatmap with colorbar normalized for each individual column. Because of
        this, pre-stimulus activity may show high values.

        .. note:: Using seaborn (sns), each row of the heatmap needs to correspond to
        each row of dictionary. If incoming data_dict is already in this format,
        you can just delete the df = df.T.

        Parameters
        ----------
        df : pd.DataFrame
            Data used in heatmap.
        axs : matplotlib.Axes object
            Custom Axes object to include.

        Returns
        -------
        images : list
            An iterable of heatmaps containing 1 column stitched together.
        """
        set_pub()
        df = df.T
        if self.sigma is not None:
            df = pd.DataFrame(gaussian_filter(df, sigma=self.sigma))
        array = df.values
        if not self.cm:
            self.cm = sns.color_palette(self.cm, n_colors=df.shape[1], as_cmap=True)
        axs = axs or plt.gca()
        axs.axis("off")
        #  Mask/hide the rest of the columns that aren't being included
        premask = np.tile(np.arange(array.shape[1]), array.shape[0]).reshape(
            array.shape
        )
        if self.title:
            axs.set_title(self.title)
        images: list = []
        for i in range(array.shape[1]):
            col = np.ma.array(array, mask=premask != i)
            im = axs.imshow(col, cmap=self.cm, aspect="auto")
            if self.save_dir:
                plt.savefig(
                    f"{self.save_dir}/){self._id}.png",
                    dpi=400,
                    bbox_inches="tight",
                    pad_inches=0.01,
                )
            images.append(im)
            plt.show()
        return images

    def single(
        self, df: pd.DataFrame,
    ):
        """
        Plot single heatmap with seaborn library.

        Parameters-
        ----------
        df : pd.DataFrame
            Data used in heatmap.
        """

        set_pub()
        df = df.copy()
        if self.sigma:
            df = pd.DataFrame(gaussian_filter(df, sigma=self.sigma))
        fig, axs = plt.subplots()
        fig.tight_layout()

        sns.heatmap(
            df, square=False, cbar=self.colorbar, cmap=self.cm, robust=self.robust,
        )
        plt.xticks([])
        if self.xlabel:
            axs.set_xlabel(self.xlabel)
        axs.set_title(self.title, fontweight="bold")
        if self.line_loc:
            axs.axvline(
                x=self.line_loc, color=self.line_color, ymin=0,
                ymax=1
            )
        if self.save_dir:
            file = f"{self.save_dir}/{self._id}.png"
            savefile = funcs.check_unique_path(file)
            plt.savefig(
                f"{savefile}", dpi=400, bbox_inches="tight", pad_inches=0.01,
            )
        plt.show()
        return fig


if __name__ == "__main__":
    heatmaps = Heatmap()
