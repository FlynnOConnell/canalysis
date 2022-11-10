#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# draw_plots.py

Module (core): Functions for drawing graphs.

"""
from __future__ import annotations

from typing import Optional, Sized

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

from canalysis.graphs.graph_utils import helpers, ax_helpers
from canalysis.graphs.base._base_figure import CalFigure
import matplotlib
helpers.update_rcparams()


def get_figure(*args, ax=None, **kwargs):
    fig = plt.figure(*args, FigureClass=CalFigure, **kwargs)
    ax = ax if ax is not None else plt.gca()
    ax.set_facecolor = "w"
    ax.patch.set_facecolor('w')
    return fig, ax


def get_axis_points(data):
    if data.shape[1] > 2:
        return [data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]]
    else:
        return [data.iloc[:, 0], data.iloc[:, 1]]


def pca_scatter(data,
                colors,
                color_dict=None,
                s: int = 10,
                angles: tuple = None):
    matplotlib.use("Qt5Agg")
    proj = '3d' if data.shape[1] > 2 else 'rectilinear'
    ax = plt.axes(projection=proj)
    ax = ax_helpers.get_axis_labels(ax, data)
    ax.set_title(f"{'Principal Components'}", fontweight="bold")
    ax.xaxis.get_label().set_fontsize(10)
    ax.yaxis.get_label().set_fontsize(10)
    ax.zaxis.get_label().set_fontsize(10)
    ax.scatter(
        *get_axis_points(data),
        c=colors,
        s=s,
        marker='o',
        facecolor='w',
    )
    if color_dict is not None:
        ax = ax_helpers.get_legend(ax, color_dict, colors, s)
    fig, ax = get_figure(ax=ax, figsize=(3, 3))
    if angles:
        ax.view_init(*angles)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.show()
    return fig


def pca_skree(variance: Sized, title: str = "") -> plt.figure:
    """
    Line chart skree plot.

    Parameters
    ----------
    variance : np.ndarray
        From PCA.explained_variance_ratio_.
    title : str, optional
        Title of graph. The default is ''.
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
        markerscale=0.4,)
    leg.get_frame().set_alpha(1)
    plt.show()
    return fig


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
