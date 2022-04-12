# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import pandas as pd
import logging
import numpy as np
from scipy.stats import zscore

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from models.SVM import SupportVectorMachine
from core.draw_plots import Plot
from core.data import CalciumData
from utils import funcs as func
from pathlib import Path

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
    'Quinine': 'red',
    'Rinse': 'lightsteelblue',
    'Lick': 'darkgray'
}

tastants = {k: colors_dict[k] for k in list(colors_dict)[:6]}
tastants_ = list(tastants.keys())

datadir = '/Users/flynnoconnell/Documents/Work/Data'
animal_id = 'PGT13'
date = '120221'
data = CalciumData(animal_id, date, datadir, pick=1)
alldata = data.alldata
t = data.taste_data.copy()
nonAS = ['MSG', 'NaCl','Sucrose', 'Citric', 'Quinine']

#%%


nt_l_d = alldata.loc[
    (alldata['MSG'] == 0 ) & 
    (alldata['NaCl'] == 0 ) & 
    (alldata['Sucrose'] == 0 ) & 
    (alldata['Citric'] == 0 ) & 
    (alldata['Quinine'] == 0) &
    (alldata['Rinse'] == 1) &
    (alldata['ArtSal'] == 1)
]
nt_l_d['colors'] = 'dodgerblue'

t_l_d = alldata.loc[
    (alldata['MSG'] == 1 ) & 
    (alldata['NaCl'] == 1 ) & 
    (alldata['Sucrose'] == 1 ) & 
    (alldata['Citric'] == 1 ) & 
    (alldata['Quinine'] == 1) &
    (alldata['Rinse'] == 0) & 
    (alldata['drylicks'] == 0)
    ]
t_l_d['colors'] = 'dodgerblue'


filt = list(nt_l_d.filter(items=data.cells))
filt.append(nt_l_d.columns[-1])
nt_l_d = nt_l_d.filter(items=filt)

t_l_d = t[t['events'].isin(nonAS)]
t_l_d.pop('Time(s)')
t_l_d.pop('events')

taste = data.taste_data

frames = pd.concat([t_l_d, nt_l_d])
frames_color = frames.pop('colors')

#%%

data.plot_stim(my_stim=['MSG', 'NaCl', 'Sucrose', 'Quinine'])
data.plot_session()

#%%

plotdata = Plot(frames, legend=tastants, colors=frames_color)
plotdata.PCA(title='{}, {}, {} Cells'.format(animal_id, date, len(data.cells)), numcomp=len(data.cells))

tasteplots = Plot(data.taste_signals, legend=tastants, colors=data.taste_colors)
tasteplots.PCA(title='{}, {}, {} Cells'.format(animal_id, date, len(data.cells)), numcomp=len(data.cells))

#%%

tastes = data.taste_signals

data.fill_taste_trials(tastes, store='outliers1')
new_df = tastes.mask((tastes - tastes.mean()).abs() > 5 * tastes.std()).dropna()


# %%

if __name__ == "__main__":
    pass     