#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# draw_plots.py

Module (core): Functions for drawing graphs.

"""
from __future__ import annotations

import logging
import webbrowser
from typing import Optional, Iterable, Any, Sized
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML
from matplotlib import rcParams, figure
import graphs.graph_utils.graph_funcs as gr_func
import matplotlib as mpl
from pathlib import Path

# mpl.use("TkAgg")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

_UNSPECIFIED = object()


def update_rcparams():
    # Function to set some easy params and avoid some annoying bugs
    rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.facecolor": "w",
            "axes.labelsize": 15,
            "lines.linewidth": 1,
            'animation.ffmpeg_path': r'/c/ffmpeg/bin/ffmpeg'
        }
    )


class CalPlot:
    """
    Base plotting class for calcium data. Contains data values and common args.
    """
    def __init__(
            self,
            data: pd.DataFrame,
            colors: Sized,
            color_dict: Optional[dict] = None,
            cmap: str = "magma",
            dpi: Optional[int] = 600,
            save_dir: str = None,
            **kwargs,
    ):
        """
        Class with graphing utilities.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Plotting data, representing points on the graph. The default is None.
        colors : Optional[Iterable], optional
            Matched colors if data values are colored.
            colors.shape must be the same as data.shape[0]. The default is None.
        cmap : str, optional
            Color for colormap. The default is 'magma'.
        save_dir : str, FileHandler, optional
            Directory for saving data. The default is None.
        **kwargs : dict
            optional arguments for matplotlib.Axes and Axes.Scatter.

        Returns
        -------
        None.
        """
        self.data: pd.DataFrame = data
        self.colors: Sized = colors
        self.color_dict: dict = color_dict
        self.cmap: str = cmap
        self.dpi: int = dpi
        self.save_dir: str = save_dir
        self.styles: dict | None = None
        self.kwargs: dict = kwargs
        update_rcparams()

    @property
    def color_dict(self, ):
        return self._color_dict

    @color_dict.setter
    def color_dict(self, new_dict):
        assert isinstance(new_dict, dict)
        self._color_dict = new_dict

    @property
    def cmap(self, ):
        return self._cmap

    @cmap.setter
    def cmap(self, new_cmap):
        assert isinstance(new_cmap, str)
        self._cmap = new_cmap

    @property
    def dpi(self, ):
        return self._dpi

    @dpi.setter
    def dpi(self, new_dpi):
        assert isinstance(new_dpi, int)
        self._dpi = new_dpi

    @property
    def save_dir(self, ):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, new_save_dir):
        if isinstance(new_save_dir, Path):
            new_save_dir = new_save_dir.__str__()
            logging.info('Save_dir converted from Type[pathlib.Path] to Type[str]')
        self._save_dir = new_save_dir

    def setstyle(self, **styles):
        for key, value in styles.items():
            setattr(self.styles, key, value)


class ScatterPlots(CalPlot):
    def get_axis_labels(self, ax):
        ax.set_xlabel(self.data.columns[0], weight="bold")
        ax.set_ylabel(self.data.columns[1], weight="bold")
        if self.data.shape[1] > 2:
            ax.set_zlabel(self.data.columns[1], weight="bold")
        return ax

    def get_axis_points(self,):
        if self.data.shape[1] > 2:
            return [self.data.iloc[:, 0], self.data.iloc[:, 1], self.data.iloc[:, 2]]
        else:
            return [self.data.iloc[:, 0], self.data.iloc[:, 1]]

    def scatterplot(
            self,
            title: Optional[str] = None,
            legend: Optional[bool] = True,
            marker: str = "o",
            savefig: Optional[bool] = False,
            msg: Optional[str] = "nomsg",
            bbox_inches: str = "tight",
            facecolor: str = "w",
            **kwargs,
    ) -> plt.figure:
        """
            Plot 2D/3D scatter plot with matplotlib.

        Parameters
        ----------
        title : Optional[str], optional
            Text to display as title. The default is None.
        legend : Optional[bool], optional
            Whether to include a legend. The default is True.
        marker : str, optional
            Shape of marker. The default is 'o'.
        savefig : Optional[bool], optional
            Alternative location to save. The default is None.
        msg : Optional[str], optional
            Message to include if save_dir is given. The default is 'nomsg'.
        bbox_inches : str, optional
            Whitespace surrounding graphs. The default is 'tight'.
        facecolor : str, optional
            Color of background. The default is 'white'.
        **kwargs : dict
            Optional inputs to matplotlib.Axes.Scatter.

        Returns
        -------
        None
        """
        numcols = '3d' if self.data.shape[1] > 2 else 'rectilinear'
        fig = plt.figure(
            figsize=(3, 3),
            frameon=False,
            facecolor=facecolor,
            edgecolor=facecolor
        )
        ax = plt.axes(projection=numcols)
        ax = self.get_axis_labels(ax)
        ax.set_facecolor = "w"
        ax.patch.set_facecolor('w')
        ax.scatter(
            *self.get_axis_points(),
            c=self.colors,
            marker=marker,
            facecolor=facecolor,
            **kwargs,
        )
        if title:
            ax.set_title(f"{title}", fontweight="bold")
        if legend:
            proxy, label = gr_func.get_handles_from_iterables(self.color_dict, self.colors)
            ax.legend(
                handles=proxy,
                labels=label,
                prop={"size": 5},
                bbox_to_anchor=(1.05, 1),
                ncol=1,
                numpoints=1,
                facecolor=ax.get_facecolor(),
                edgecolor='w'
            )
        plt.tight_layout()
        plt.show()
        if savefig:
            fig.savefig(
                self.save_dir + f"/{title}_{msg}.png",
                bbox_inches=bbox_inches,
                dpi=self.dpi,
                facecolor=fig.get_facecolor(),
                edgecolor=None
            )
        return fig

    @staticmethod
    def skree(variance: Sized, title: str = "") -> plt.figure:
        """
        Line chart skree plot.

        Parameters
        ----------
        variance : np.ndarray
            From PCA.explained_variance_ratio_.
        title : str, optional
            Title of graph. The default is ''.

        Returns
        -------
        None
            DESCRIPTION.
        """
        lab = np.arange(len(variance)) + 1
        fig, ax = plt.subplot(111)
        ax.plot(lab, variance, "o-", linewidth=2, color="blue")
        ax.set_xticks(np.arange(0, lab[-1], 1.0))
        ax.set_title(f"{title}" + "Scree Plot")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained (%)")
        leg = plt.legend(
            ["Eigenvalues from SVD"],
            loc="best",
            borderpad=0.3,
            shadow=False,
            prop=fm.FontProperties(size="small"),
            markerscale=0.4,
        )
        leg.get_frame().set_alpha(1)
        plt.show()
        return fig

    def scatter_animated(
            self,
            url: str = r"C:\Users\flynn\Desktop\figs\temp.mp4",
            size: int = 5,
            marker: str = "o",
            alpha: int = 1,
    ) -> HTML:
        """
            Animated 3D Scatter plot.
            Plot gets saved to a temporary html file (location provided by save_dir).
            Args:
                url (str): Save location.
                size (int): Size of graph markers. Default = 5.
                marker (str): Shape of marker. Default is circle.
                alpha (int): Alpha of markers.

            Returns:
                None
        """
        fig = plt.figure()
        ax = fig.gca(projection="3d")

        numcols = '3d' if self.data.shape[1] > 2 else 'rectilinear'
        fig = plt.figure(dpi=self.dpi, frameon=True, facecolor='white', projection=numcols)
        ax = plt.axes(projection=numcols)
        ax = self.get_axis_labels(ax)
        ax.set_title("3D - PCA")
        ax.scatter(
            *self.get_axis_points(),
            c=self.colors,
            s=size, marker=marker, alpha=alpha)

        proxy, label = gr_func.get_handles_from_dict(self.color_dict)
        ax.legend(
            handles=proxy,
            labels=label,
            loc="upper right",
            prop={"size": 6},
            bbox_to_anchor=(1, 1),
            ncol=2,
            numpoints=1,
        )

        def init():
            ax.plot(*self.get_axis_points(), linewidth=0, antialiased=False)
            return (fig,)

        def animate(i):
            ax.view_init(elev=30.0, azim=3.6 * i)
            return (fig,)

        ani = animation.FuncAnimation(
            fig, animate, init_func=init, frames=400, interval=100, blit=True
        )

        data = HTML(ani.to_html5_video())
        with open(f'{url}', "wb") as f:
            f.write(data.data.encode("UTF-8"))
        webbrowser.open(url, new=2)
        return data

    @staticmethod
    def confusion_matrix(
            y_pred,
            y_true,
            labels: list,
            xaxislabel: Optional[str] = None,
            yaxislabel: Optional[str] = None,
            caption: Optional[str] = "",
    ) -> np.array:
        """

        Parameters
        ----------
        y_pred : TYPE
            DESCRIPTION.
        y_true : TYPE
            DESCRIPTION.
        labels : list
            DESCRIPTION.
        xaxislabel : Optional[str], optional
            DESCRIPTION. The default is None.
        yaxislabel : Optional[str], optional
            DESCRIPTION. The default is None.
        caption : Optional[str], optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        mat : TYPE
            DESCRIPTION.

        """
        from sklearn.metrics import confusion_matrix
        mat = confusion_matrix(y_pred, y_true)
        sns.heatmap(
            mat.T,
            square=True,
            annot=True,
            fmt="d",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
        )
        if xaxislabel:
            plt.xlabel(xaxislabel)
        else:
            plt.xlabel("true label")
        if yaxislabel:
            plt.ylabel(yaxislabel)
        else:
            plt.ylabel("predicted label")
        if caption:
            plt.text(0, -0.03, caption, fontsize="small")
        plt.show()
        return mat
