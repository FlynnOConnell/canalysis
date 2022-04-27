#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# heatmaps.py
"""

from __future__ import annotations
from typing import Iterable, Optional
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import numpy as np
from core.calciumdata import CalciumData
from core.taste_data import TasteData
import matplotlib.pyplot as plt
from matplotlib import rcParams


def set_pub():
    """Update matplotlib backend default styles to be bigger and bolder."""
    rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": 'bold',
        'axes.facecolor': 'w',
        "axes.labelsize": 15,
        "lines.linewidth": 1,
    })

#%%


class Heatmap(object):
    # Initialize the attributes to apply to our heatmaps, nearly all optional and can be ignored.
    def __init__(self,
                 save_dir: str | None = '',
                 cm: str = 'magma',
                 _id: str | None = '',
                 title: str | None = '',
                 sigma: int | None = None,
                 square: bool = False,
                 colorbar: bool = False,
                 robust: bool = False,
                 line_loc: Optional[int] = 0,
                 line_width: Optional[int] = 3,
                 line_color: str = 'white'
                 ):
        self.save_dir = save_dir
        self.cm = plt.get_cmap(cm)
        self._id = _id
        self.title = title
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
            Identifyer that will be appended to the filenames this batch of graphs.
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

    def nested(self,
               data_dict: dict,
               **axargs):
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
            ### Plot data
            fig, axs = plt.subplots()
            sns.heatmap(df, square=self.square, cbar=self.colorbar, robust=self.robust, **axargs)
            axs.axis('off')
            if self.line_loc:
                axs.axvline(x=self.line_loc, color=self.line_color, linewidth=self.line_width)
            if self.save_dir:
                plt.savefig(f'{self.save_dir}/{self._id}.png',
                            dpi=400, bbox_inches='tight', pad_inches=0.01)
            return fig, axs

    def columnwise(self,
                   data: pd.DataFrame,
                   axs: object = None,
                   ):
        """
        Plot heatmap with colorbar normalized for each individual column. Because of this, pre-stimulus
        activity may show high values.

        .. note::
            Using seaborn (sns), each row of the heatmap needs to correspond to each row of dictionary.
            If incoming data_dict is already in this format, you can just delete the df = df.T.

        Parameters
        ----------
        data : pd.DataFrame
            Data used in heatmap.
        axs : matplotlib.Axes object
            Custom Axes object to include.

        Returns
        -------
        images : list
            An iterable of heatmaps containing 1 column stitched together.

        """
        set_pub()
        df = data.T
        if self.sigma is not None:
            df = pd.DataFrame(gaussian_filter(data, sigma=self.sigma))
        # We need to grab each column individually
        array = df.values
        # Fetch a pretty color palete:
        c = sns.color_palette(self.cm, n_colors=df.shape[1], as_cmap=True)
        # Plot data
        axs = axs or plt.gca()
        axs.axis('off')
        # We need to mask/hide the rest of the columns that aren't being included
        premask = np.tile(np.arange(array.shape[1]), array.shape[0]).reshape(array.shape)
        if self.title:
            axs.set_title(self.title)
        images: list = []
        for i in range(array.shape[1]):
            col = np.ma.array(array, mask=premask != i)
            im = axs.imshow(col, cmap=self.cm, aspect='auto')
            if self.save_dir:
                plt.savefig(f'{self.save_dir}/){self._id}.png',
                            dpi=400, bbox_inches='tight', pad_inches=0.01)
            images.append(im)
        return images

    def single(self,
               data: pd.DataFrame,
               ):
        """
        Plot single heatmap with seaborn library.

        Parameters
        ----------
        data : pd.DataFrame
            Data used in heatmap.
        """
        set_pub()
        data = data.copy()
        if self.sigma:
            data = pd.DataFrame(gaussian_filter(data, sigma=self.sigma))
        fig, axs = plt.subplots()
        sns.heatmap(data, square=self.square, cbar=self.colorbar, cmap=self.cm, robust=self.robust)
        axs.axis('off')
        if self.line_loc:
            axs.axvline(x=self.line_loc, color=self.line_color, linewidth=self.line_width)
        if self.save_dir:
            plt.savefig(f'{self.save_dir}/{self._id}.png',
                        dpi=400, bbox_inches='tight', pad_inches=0.01)
        plt.show()
        return fig, axs


# %% Data Getters

datadir = 'A:\\'
animal = 'PGT13'
date = '121021'
data = CalciumData(datadir, animal, date)
#%%
taste = TasteData(data.zscores,data.timestamps,data.color_dict)
hm = taste.tastedata['Lick'].reset_index(drop=True).iloc[0:40].drop(['Time(s)', 'colors', 'events'], axis=1)
#%%
heatmaps = Heatmap()

heatmaps.single(hm)
heatmaps.columnwise(hm)


# # %%
# animal2 = 'PGT08'
# dates2 = ['071621', '072721']
#
# animal3 = 'PGT06'
# dates3 = ['051321', '050721']
#
# animal1_data = []
# for date in dates:
#     data_day = CalciumData(animal1, date, datadir)
#     print((data_day.session, data_day.cells, data_day.numlicks))
#     animal1_data.append(data_day)
# cells1 = animal1_data[0].cells
#
# animal2_data = []
# for date in dates2:
#     data_day = CalciumData(animal2, date, datadir)
#     print((data_day.session, data_day.cells, data_day.numlicks))
#     animal2_data.append(data_day)
# cells2 = animal2_data[0].cells
#
# animal3_data = []
# for date in dates3:
#     data_day = CalciumData(animal3, date, datadir)
#     print((data_day.session, data_day.cells, data_day.numlicks))
#     animal3_data.append(data_day)
# cells3 = animal3_data[0].cells
#
# # %% Fill Data Containers
#
# # Animal 1 Day 1
# as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
# s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
# n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
# ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
# q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
# msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
#
# t_as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
# t_s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
# t_n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
# t_ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
# t_q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
# t_msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
#
# tt_as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
# tt_s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
# tt_n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
# tt_ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
# tt_q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
# tt_msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
#
# day1 = []
#
# s_day1 = pd.concat([pd.concat([v for k, v in s_zdict_day1.items()]),
#                     pd.concat([v for k, v in t_s_zdict_day1.items()]),
#                     pd.concat([v for k, v in tt_s_zdict_day1.items()])],
#                    axis=0)
# day1.append(s_day1)
#
# s_day1.reset_index(drop=True, inplace=True)
# s_day1['value'] = s_day1[49] - s_day1[0]
# s_day1.sort_values(by='value', inplace=True, ascending=False)
# s_day1.drop(columns='value', inplace=True)
# s_day1.dropna(axis=1, inplace=True)
#
# as_day1 = pd.concat([pd.concat([v for k, v in as_zdict_day1.items()]),
#                      pd.concat([v for k, v in t_as_zdict_day1.items()]),
#                      pd.concat([v for k, v in tt_as_zdict_day1.items()])],
#                     axis=0).reset_index(drop=True).reindex(s_day1.index)
#
# day1.append(as_day1)
#
# n_day1 = pd.concat([pd.concat([v for k, v in n_zdict_day1.items()]),
#                     pd.concat([v for k, v in t_n_zdict_day1.items()]),
#                     pd.concat([v for k, v in tt_n_zdict_day1.items()])],
#                    axis=0).reset_index(drop=True).reindex(s_day1.index)
# day1.append(n_day1)
#
# ca_day1 = pd.concat([pd.concat([v for k, v in ca_zdict_day1.items()]),
#                      pd.concat([v for k, v in t_ca_zdict_day1.items()]),
#                      pd.concat([v for k, v in tt_ca_zdict_day1.items()])],
#                     axis=0).reset_index(drop=True).reindex(s_day1.index)
#
# day1.append(ca_day1)
#
# q_day1 = pd.concat([pd.concat([v for k, v in q_zdict_day1.items()]),
#                     pd.concat([v for k, v in t_q_zdict_day1.items()]),
#                     pd.concat([v for k, v in tt_q_zdict_day1.items()])],
#                    axis=0).reset_index(drop=True).reindex(s_day1.index).dropna()
# day1.append(q_day1)
#
# msg_day1 = pd.concat([pd.concat([v for k, v in msg_zdict_day1.items()]),
#                       pd.concat([v for k, v in t_msg_zdict_day1.items()]),
#                       pd.concat([v for k, v in tt_msg_zdict_day1.items()])],
#                      axis=0).reset_index(drop=True).reindex(s_day1.index).dropna()
#
# as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
# s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
# n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
# ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
# q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
# msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
#
# t_as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
# t_s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
# t_n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
# t_ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
# t_q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
# t_msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
#
# tt_as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
# tt_s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
# tt_n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
# tt_ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
# tt_q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
# tt_msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
#
# s_day2 = pd.concat([pd.concat([v for k, v in s_zdict_day2.items()]),
#                     pd.concat([v for k, v in t_s_zdict_day2.items()]),
#                     pd.concat([v for k, v in tt_s_zdict_day2.items()])],
#                    axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
#
# as_day2 = pd.concat([pd.concat([v for k, v in as_zdict_day2.items()]),
#                      pd.concat([v for k, v in t_as_zdict_day2.items()]),
#                      pd.concat([v for k, v in tt_as_zdict_day2.items()])],
#                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
#
# n_day2 = pd.concat([pd.concat([v for k, v in n_zdict_day2.items()]),
#                     pd.concat([v for k, v in t_n_zdict_day2.items()]),
#                     pd.concat([v for k, v in tt_n_zdict_day2.items()])],
#                    axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
#
# ca_day2 = pd.concat([pd.concat([v for k, v in ca_zdict_day2.items()]),
#                      pd.concat([v for k, v in t_ca_zdict_day2.items()]),
#                      pd.concat([v for k, v in tt_ca_zdict_day2.items()])],
#                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
# q_day2 = pd.concat([pd.concat([v for k, v in q_zdict_day2.items()]),
#                     pd.concat([v for k, v in t_q_zdict_day2.items()]),
#                     pd.concat([v for k, v in tt_q_zdict_day2.items()])],
#                    axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
# msg_day2 = pd.concat([pd.concat([v for k, v in msg_zdict_day2.items()]),
#                       pd.concat([v for k, v in t_msg_zdict_day2.items()]),
#                       pd.concat([v for k, v in tt_msg_zdict_day2.items()])],
#                      axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
#
# day1 = []
# day1.extend([as_day1, s_day1, n_day1, q_day1, msg_day1])
