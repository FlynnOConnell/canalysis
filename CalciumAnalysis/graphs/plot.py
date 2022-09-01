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

from graph_utils import helpers, ax_helpers
from graph_utils.cafigure import CalFigure
import matplotlib

matplotlib.use("TkAgg")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

# Globals
_UNSPECIFIED = object()


def get_figure(*args, ax=None, **kwargs):
    helpers.update_rcparams()
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


def pca_scatter(data, colors, color_dict=None, s: int = 10, **kwargs):
    proj = '3d' if data.shape[1] > 2 else 'rectilinear'
    ax = plt.axes(projection=proj)
    ax = ax_helpers.get_axis_labels(ax, data)
    ax.set_title(f"{'Principal Components'}", fontweight="bold")
    ax.scatter(
        *get_axis_points(data),
        c=colors,
        s=s,
        marker='o',
        facecolor='w',
        **kwargs,)
    if color_dict is not None:
        ax = ax_helpers.get_legend(ax, color_dict, colors, s)
    fig, ax = get_figure(ax=ax, figsize=(3, 3))
    plt.show()
    return fig, ax


def pca_skree(variance: Sized, title: str = "") -> plt.figure:
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
