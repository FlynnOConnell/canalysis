#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:24:43 2022

@author: flynnoconnell
"""

from __future__ import annotations
from typing import Tuple, Iterable, Optional, Sized, Any, Mapping
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import numpy as np
import logging
import scipy.stats as stats
from core.data import CalciumData
from graphs.draw_plots import Plot
from utils import funcs as func
from utils import stats_helpers as stat_help

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


def sns_heatmap(dct, tastant='',plottype='', sigma=2,
                square=False,cbar=False,y=30,linewidth=3,
                color='white',save=False,robust=False,
                **axargs):

    for cell, df in dct.items():
        df = df.T
        df_smooth = pd.DataFrame(gaussian_filter(df, sigma=sigma))
        # cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)
        ### Plot data
        fig, ax = plt.subplots()
        ax = sns.heatmap(df_smooth,square=square, cbar=cbar,robust=robust, **axargs)
        ax.axis('off')
        # ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
        ax.axhline(y=y, color=color, linewidth=linewidth)
        if save:
            plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{cell}_{tastant}_{plottype}.png',
                        dpi=400, bbox_inches='tight', pad_inches=0.01)
            
def single_sns_heatmap(df, tastant='',plottype='', sigma=2,
                square=False,cbar=False,x=10,linewidth=3,
                color='white',save=False,robust=False,
                **axargs):
    if sigma:
        df = pd.DataFrame(gaussian_filter(df, sigma=sigma))
    # cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)
    ### Plot data
    fig, ax = plt.subplots()
    ax = sns.heatmap(df, square=square, cbar=cbar, robust=robust, **axargs)
    ax.axis('off')
    # ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
    ax.axvline(x=x, color=color, linewidth=linewidth)
    if save:
        plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{tastant}_{plottype}.png',
                    dpi=400, bbox_inches='tight', pad_inches=0.01)

#%% Data


color_dict = {
    'ArtSal': 'dodgerblue',
    'MSG': 'darkorange',
    'NaCl': 'lime',
    'Sucrose': 'magenta',
    'Citric': 'yellow',
    'Quinine': 'red',
    'Rinse': 'lightsteelblue',
    'Lick': 'darkgray'
}

pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')
datadir = '/Users/flynnoconnell/Documents/Work/Data'

animal1 = 'PGT13'
dates = ['121021', '121721']

animal2 = 'PGT08'
dates2 = ['071621', '072721']

animal3 = 'PGT06'
dates3 = ['051321', '050721']

animal1_data = []
for date in dates:
    data_day = CalciumData(animal1, date, datadir, pick=0,color_dict=color_dict)
    print((data_day.session, data_day.cells, data_day.numlicks))
    animal1_data.append(data_day)
cells1 = animal1_data[0].cells

animal2_data = []
for date in dates2:
    data_day = CalciumData(animal2, date, datadir, pick=0,color_dict=color_dict)
    print((data_day.session, data_day.cells, data_day.numlicks))
    animal2_data.append(data_day)
cells2 = animal2_data[0].cells

animal3_data = []
for date in dates3:
    data_day = CalciumData(animal3, date, datadir, pick=0,color_dict=color_dict)
    print((data_day.session, data_day.cells, data_day.numlicks))
    animal3_data.append(data_day)
cells3 = animal3_data[0].cells



#%% Fill Data Containers

# Animal 1 Day 1
as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells1, [animal1_data[0]], do_zscore=True, baseline=1)

t_as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells2, [animal2_data[0]], do_zscore=True, baseline=1)

tt_as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells3, [animal3_data[0]], do_zscore=True, baseline=1)

day1 = []

s_day1 = pd.concat([pd.concat([v for k,v in s_zdict_day1.items()]),
                     pd.concat([v for k,v in t_s_zdict_day1.items()]),
                     pd.concat([v for k,v in tt_s_zdict_day1.items()])],
                     axis=0)
day1.append(s_day1)

s_day1.reset_index(drop=True, inplace=True)
s_day1['value'] = s_day1[49] - s_day1[0]
s_day1.sort_values(by='value', inplace=True, ascending=False)
s_day1.drop(columns='value', inplace=True)
s_day1.dropna(axis=1, inplace=True)

as_day1 = pd.concat([pd.concat([v for k,v in as_zdict_day1.items()]),
                     pd.concat([v for k,v in t_as_zdict_day1.items()]),
                     pd.concat([v for k,v in tt_as_zdict_day1.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index)

day1.append(as_day1)


n_day1 = pd.concat([pd.concat([v for k,v in n_zdict_day1.items()]),
                     pd.concat([v for k,v in t_n_zdict_day1.items()]),
                     pd.concat([v for k,v in tt_n_zdict_day1.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index)
day1.append(n_day1)


ca_day1 = pd.concat([pd.concat([v for k,v in ca_zdict_day1.items()]),
                     pd.concat([v for k,v in t_ca_zdict_day1.items()]),
                     pd.concat([v for k,v in tt_ca_zdict_day1.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index)

day1.append(ca_day1)

q_day1 = pd.concat([pd.concat([v for k,v in q_zdict_day1.items()]),
                     pd.concat([v for k,v in t_q_zdict_day1.items()]),
                     pd.concat([v for k,v in tt_q_zdict_day1.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna()
day1.append(q_day1)


msg_day1 = pd.concat([pd.concat([v for k,v in msg_zdict_day1.items()]),
                     pd.concat([v for k,v in t_msg_zdict_day1.items()]),
                     pd.concat([v for k,v in tt_msg_zdict_day1.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna()

as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells1, [animal1_data[1]], do_zscore=True, baseline=1)

t_as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells2, [animal2_data[1]], do_zscore=True, baseline=1)

tt_as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells3, [animal3_data[1]], do_zscore=True, baseline=1)

s_day2 = pd.concat([pd.concat([v for k,v in s_zdict_day2.items()]),
                     pd.concat([v for k,v in t_s_zdict_day2.items()]),
                     pd.concat([v for k,v in tt_s_zdict_day2.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')

# s_day2.reset_index(drop=True, inplace=True)
# s_day2['value'] = s_day2[49] - s_day2[0]
# s_day2.sort_values(by='value', inplace=True, ascending=False)
# s_day2.drop(columns='value', inplace=True)
# s_day2.dropna(axis=1, inplace=True)

as_day2 = pd.concat([pd.concat([v for k,v in as_zdict_day2.items()]),
                     pd.concat([v for k,v in t_as_zdict_day2.items()]),
                     pd.concat([v for k,v in tt_as_zdict_day2.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')

n_day2 = pd.concat([pd.concat([v for k,v in n_zdict_day2.items()]),
                     pd.concat([v for k,v in t_n_zdict_day2.items()]),
                     pd.concat([v for k,v in tt_n_zdict_day2.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')

ca_day2 = pd.concat([pd.concat([v for k,v in ca_zdict_day2.items()]),
                     pd.concat([v for k,v in t_ca_zdict_day2.items()]),
                     pd.concat([v for k,v in tt_ca_zdict_day2.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
q_day2 = pd.concat([pd.concat([v for k,v in q_zdict_day2.items()]),
                     pd.concat([v for k,v in t_q_zdict_day2.items()]),
                     pd.concat([v for k,v in tt_q_zdict_day2.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
msg_day2 = pd.concat([pd.concat([v for k,v in msg_zdict_day2.items()]),
                     pd.concat([v for k,v in t_msg_zdict_day2.items()]),
                     pd.concat([v for k,v in tt_msg_zdict_day2.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')




day1 = []
day1.extend([as_day1, s_day1, n_day1, q_day1, msg_day1])






#%% HEATMAPS

def columnwise_heatmap2(df, ax=None, plottype='', tastant='', square=True,
                       x=10, linecolor='white', linewidth='4',
                       cm='magma',aspect='auto', sigma=2, savefig=False,
                       **kw):
    
    df = df.T
    df_smooth = pd.DataFrame(gaussian_filter(df, sigma=sigma))
    array = df_smooth.values
    c = sns.color_palette(cm, n_colors = df_smooth.shape[1], as_cmap = True)

    ### Plot data
    ax = ax or plt.gca()
    ax.axis('off')
    premask = np.tile(np.arange(array.shape[1]), array.shape[0]).reshape(array.shape)
    
    plottype = plottype
    tastant=tastant
    images = []
    for i in range(array.shape[1]):
        col = np.ma.array(array, mask = premask != i)
        # im = sns.heatmap(col, cmap=c, square=square, cbar=False)
        im = ax.imshow(col, cmap=c, aspect=aspect)
        images.append(im)
        
    return images


#%%

def single_sns_heatmap(df, tastant='',plottype='', sigma=2,
                square=False,cbar=False, x=10, linewidth=3,
                color='white',cmap='magma', save=False,robust=False,dpi=400,
                **axargs):
    
    if sigma:
        df = pd.DataFrame(gaussian_filter(df, sigma=sigma))
    # cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)
    ### Plot data
    fig, ax = plt.subplots()
    ax = sns.heatmap(df, square=square, cbar=cbar, cmap=cmap, robust=robust, **axargs)
    
    ax.axis('off')
    ax.axvline(x=x, color=color, linewidth=linewidth)
    if save:
        plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{tastant}.png',
                    dpi=dpi, bbox_inches='tight', pad_inches=0.01)


for ind, day in enumerate(day1):
    single_sns_heatmap(day,
                tastant='q',
                plottype='zscore',
                sigma=None,
                dpi=600,
                square=True,
                cmap='magma',
                cbar=True,
                x=10,
                linewidth=3,
                color='white',
                robust=True,
                save=False)


#%%

df_smooth = pd.DataFrame(gaussian_filter(df, sigma=2))
cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)

### Plot data
fig, ax = plt.subplots(figsize=(8,4.5))
ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
ax.axvline(x=10, color='white', linewidth=4)
ty = 'zscore'
tastant='AS'
plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{tastant}_{ty}.png',
            dpi=400, bbox_inches='tight', pad_inches=0.01)








sns_heatmap(as_day1,
            tastant='S',
            plottype='zscore',
            sigma=2,
            square=True,
            cbar=False,
            y=30,
            linewidth=3,
            color='white',
            robust=False,
            save=False)

sns_heatmap(N_zscore_dict,
            tastant='N',
            plottype='zscore',
            sigma=2,
            square=True,
            cbar=False,
            y=30,
            linewidth=3,
            color='white',
            robust=False,
            save=True)

sns_heatmap(CA_zscore_dict,
            tastant='CA',
            plottype='zscore',
            sigma=2,
            square=True,
            cbar=False,
            y=30,
            linewidth=3,
            color='white',
            robust=False,
            save=True)

sns_heatmap(Q_zscore_dict,
            tastant='Q',
            plottype='zscore',
            sigma=2,
            square=True,
            cbar=False,
            y=30,
            linewidth=3,
            color='white',
            robust=False,
            save=True)

sns_heatmap(MSG_zscore_dict,
            tastant='MSG',
            plottype='zscore',
            sigma=2,
            square=True,
            cbar=False,
            y=30,
            linewidth=3,
            color='white',
            robust=False,
            save=True)
