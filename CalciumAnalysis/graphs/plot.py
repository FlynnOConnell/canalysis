#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# draw_plots.py

Module (core): Functions for drawing graphs.

"""
from __future__ import annotations

import logging

from typing import Optional, Sized
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams, figure
import graphs.graph_utils.graph_funcs as gr_func
import matplotlib
from pathlib import Path

matplotlib.use("TkAgg")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

# Globals
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
            'animation.ffmpeg_path': r'/c/ffmpeg/bin/ffmpeg',
            'scatter.edgecolors': None
        })


def get_figure(*args, ax=None, **kwargs):
    update_rcparams()
    fig = plt.figure(FigureClass=CalFigure, *args, **kwargs)
    ax = ax if ax is not None else plt.gca()
    ax.set_facecolor = "w"
    ax.patch.set_facecolor('w')
    return fig, ax


def get_axis_points(data):
    if data.shape[1] > 2:
        return [data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]]
    else:
        return [data.iloc[:, 0], data.iloc[:, 1]]


def get_axis_labels(ax, data):
    ax.set_xlabel(data.columns[0], weight="bold")
    ax.set_ylabel(data.columns[1], weight="bold")
    if data.shape[1] > 2:
        ax.set_zlabel(data.columns[1], weight="bold")
    return ax


def get_legend(ax, color_dict, colors):
    mydict = {k: v for k, v in color_dict.items() if color_dict[k] in np.unique(colors)}
    proxy, label = gr_func.get_handles_from_dict(mydict, colors)
    ax.legend(
        handles=proxy,
        labels=label,
        prop={"size": 5},
        bbox_to_anchor=(1.05, 1),
        ncol=1,
        numpoints=1,
        facecolor=ax.get_facecolor(),
        edgecolor='w')
    return ax


def pca_scatter(data, colors, color_dict=None, **kwargs):
    proj = '3d' if data.shape[1] > 2 else 'rectilinear'
    ax = plt.axes(projection=proj)
    ax = get_axis_labels(ax, data)
    ax.set_title(f"{'Principal Components'}", fontweight="bold")
    ax.scatter(
        *get_axis_points(data),
        c=colors,
        marker='o',
        facecolor='w',
        **kwargs, )
    if color_dict is not None:
        ax = get_legend(ax, color_dict, colors)
    fig, ax = get_figure(ax=ax, figsize=(3, 3))
    plt.show()
    return fig, ax


class CalFigure(matplotlib.figure.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.set_dpi(dpi)
        self.tight_layout()
        self.frameon = False

    def save(self, title='', *args, **kwargs):
        self.savefig(
            f"{title}.png",
            bbox_inches='tight',
            facecolor=self.get_facecolor(),
            edgecolor=None,
            *args,
            **kwargs)


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
        markerscale=0.4, )
    leg.get_frame().set_alpha(1)
    plt.show()
    return fig


# def scatter_animated(
#         url: str = r"C:\Users\flynn\Desktop\figs\temp.mp4",
#         size: int = 5,
#         marker: str = "o",
#         alpha: int = 1,
# ) -> HTML:
#     """
#         Animated 3D Scatter plot.
#         Plot gets saved to a temporary html file (location provided by save_dir).
#         Args:
#             url (str): Save location.
#             size (int): Size of graph markers. Default = 5.
#             marker (str): Shape of marker. Default is circle.
#             alpha (int): Alpha of markers.
#
#         Returns:
#             None
#     """
#     fig = plt.figure()
#     ax = fig.gca(projection="3d")
#
#     numcols = '3d' if data.shape[1] > 2 else 'rectilinear'
#     fig = plt.figure(dpi=dpi, frameon=True, facecolor='white', projection=numcols)
#     ax = plt.axes(projection=numcols)
#     ax = get_axis_labels(ax)
#     ax.set_title("3D - PCA")
#     ax.scatter(
#         *get_axis_points(),
#         c=colors,
#         s=size, marker=marker, alpha=alpha)
#
#     proxy, label = gr_func.get_handles_from_dict(self.color_dict)
#     ax.legend(
#         handles=proxy,
#         labels=label,
#         loc="upper right",
#         prop={"size": 6},
#         bbox_to_anchor=(1, 1),
#         ncol=2,
#         numpoints=1,
#     )
#
#     def init():
#         ax.plot(*self.get_axis_points(), linewidth=0, antialiased=False)
#         return (fig,)
#
#     def animate(i):
#         ax.view_init(elev=30.0, azim=3.6 * i)
#         return (fig,)
#
#     ani = animation.FuncAnimation(
#         fig, animate, init_func=init, frames=400, interval=100, blit=True
#     )
#
#     data = HTML(ani.to_html5_video())
#     with open(f'{url}', "wb") as f:
#         f.write(data.data.encode("UTF-8"))
#     webbrowser.open(url, new=2)
#     return data


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
