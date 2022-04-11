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

datadir = '/Users/flynnoconnell/Documents/Work/Data'
animal_id = 'PGT13'
date = '120221'

data = CalciumData(animal_id, date, datadir, pick=1)

#%%

data.plot_stim(my_stim=['MSG', 'NaCl', 'Sucrose', 'Quinine'])
data.plot_session()

#%%

fig = Plot(data.taste_signals, legend=data.tastant_colors_dict, colors=data.taste_colors)
fig.PCA(title='{}, {}, {} Cells'.format(animal_id, date, len(data.cells)), numcomp=2)


#%%

tastes = data.taste_signals

data.fill_taste_trials(tastes, store='outliers1')
new_df = tastes.mask((tastes - tastes.mean()).abs() > 5 * tastes.std()).dropna()


# %%

if __name__ == "__main__":
    pass     