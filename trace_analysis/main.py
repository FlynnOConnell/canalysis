# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import pandas as pd
import logging
import numpy as np

from graphs.draw_plots import Plot
from core.data import CalciumData
from utils import funcs as func
from core.draw_plots import set_pub

pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')

# %% Initialize data
colors_dict = {
    'ArtSal': 'dodgerblue',
    'MSG': 'darkorange',
    'NaCl': 'lime',
    'Sucrose': 'magenta',
    'Citric': 'yellow',
    'Quinine': 'red'
    # 'Rinse': 'lightsteelblue',
    # 'drylick': 'darkgray'
}

tastants = {k: colors_dict[k] for k in list(colors_dict)[:6]}

datadir = '/Users/flynnoconnell/Documents/Work/Data'
animal_id = 'PGT08'
date = '070121'

data = CalciumData(animal_id, date, datadir, pick=0)
taste_df = data.taste_df
lick_df = data.lick_df
nonlick_df = data.nonlick_df
colors = taste_df.pop('colors')

#%%
# df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
set_pub()

df = pd.concat([data.lick_df, data.nonlick_df])
colors = df['colors']

#%% Get PCA Plots
df.drop(columns='colors', inplace=True)

plotdata = Plot(taste_df, legend=colors_dict, colors=colors)
plotdata.PCA(
    title=f'{animal_id}, {date}, Lick vs Not Licking',
    numcomp=2,
    size=15,
    norm=False,
    noscale=True,
    ss=True,
    rs=True,
    mm=True,
    remove_outliers=False,
    std=4,
    skree=False
    )

#%%
set_pub()

plotdata = Plot(df, legend=colors_dict, colors=colors)
plotdata.scatter_3d(
    title=f'{animal_id}, {date}, Lick vs Not Licking'
    )

#%%
set_pub()
Plot.plot_session(data.cells,
                  data.signals,
                  data.time,
                  data.session,
                  data.numlicks,
                  data.timestamps,
                  save_dir=True
                  )

if __name__ == "__main__":
    pass     