#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#graph_funcs.py

Module (graph): General functions for graphing.
"""
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from matplotlib import lines


def get_handles(color_dict: dict,
                marker: Optional[str] = None,
                linestyle: Optional[int] = 'none',
                **kwargs
                ) -> Tuple[list, list]:
    """
    Get matplotlib handles for input dictionary.

    Args:
        color_dict (dict): Dictionary of event:color k/v pairs.
        linestyle (str): Connecting lines, default = none.
        marker (str): Shape of scatter point, default is circle.

    Returns:
        proxy (list): matplotlib.lines.line2D appended list.
        label (list): legend labels for each proxy.

    """

    proxy, label = [], []
    for t, c in color_dict.items():
        proxy.append(lines.Line2D([0], [0], marker=marker, markerfacecolor=c, linestyle=linestyle, **kwargs))
        label.append(t)
    return proxy, label


def confidence_ellipse(x, y, ax, n_std=1.8, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
            
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    facecolor : str
        Color of background.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x 
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y 
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
