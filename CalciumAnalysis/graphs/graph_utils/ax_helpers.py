#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ax_helpers.py

Module (graphs.graph_utils): Functions for manipulating axes objects.

"""
from __future__ import annotations

import logging
import numpy as np
import graphs.graph_utils.helpers as gr_func


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


def get_axis_labels(ax, data):
    ax.set_xlabel(data.columns[0], weight="bold")
    ax.set_ylabel(data.columns[1], weight="bold")
    if data.shape[1] > 2:
        ax.set_zlabel(data.columns[1], weight="bold")
    return ax


def get_legend(ax, color_dict, colors, markersize):
    mydict = {k: v for k, v in color_dict.items() if color_dict[k] in np.unique(colors)}
    proxy, label = gr_func.get_handles_from_dict(mydict, markersize)
    ax.legend(
        handles=proxy,
        labels=label,
        prop={"size": 20},
        bbox_to_anchor=(1.05, 1),
        ncol=2,
        numpoints=1,
        facecolor=ax.get_facecolor(),
        edgecolor=None,
        fancybox=True,
        markerscale=True)
    return ax
