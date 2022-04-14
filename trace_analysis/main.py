# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import importlib
import pandas as pd
import logging
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats

import draw_plots
from core.data import CalciumData
from core.draw_plots import Plot
from scipy.ndimage.filters import gaussian_filter

importlib.reload(draw_plots)
#%%
pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')

datadir = 'A:\\'

# animal_id = 'PGT13'
# dates = ['011222', '011422', '011922', '030322', '032422', '120221', '121021', '121721']

animal_id = 'PGT08'
dates = ['060221', '062321', '071621', '072721', '082021', '092321', '093021']
# '060221',
alldata = []
for date in dates:
    data_day = CalciumData(animal_id, date, datadir, pick=0)
    print((data_day.session, data_day.cells, data_day.numlicks))
    alldata.append(data_day)

cells = alldata[0].cells

# %%
AS_zscore_dict = {}
S_zscore_dict = {}
N_zscore_dict = {}
CA_zscore_dict = {}
Q_zscore_dict = {}
MSG_zscore_dict = {}

AS_dict = {}
S_dict = {}
N_dict = {}
CA_dict = {}
Q_dict = {}
MSG_dict = {}
#%% AS
for ind, cell in enumerate(cells):
    data_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['ArtSal']:
                for iteration, trial in enumerate(times):
                    # Index to analysis window
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    data_holder = pd.concat([data_holder, signal], axis=1)

    AS_dict[cell] = data_holder
    AS_zscore_dict[cell] = zscore_holder

for cell, cell_df in S_dict.items():
    AS_dict[cell] = cell_df.T
for cell, cell_df in S_zscore_dict.items():
    AS_zscore_dict[cell] = cell_df.T


# %% Sucrose
for ind, cell in enumerate(cells):
    data_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['Sucrose']:
                for iteration, trial in enumerate(times):
                    # Index to analysis window
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    data_holder = pd.concat([data_holder, signal], axis=1)

    S_dict[cell] = data_holder
    S_zscore_dict[cell] = zscore_holder

for cell, cell_df in S_dict.items():
    S_dict[cell] = cell_df.T
for cell, cell_df in S_zscore_dict.items():
    S_zscore_dict[cell] = cell_df.T

# %% NaCl

for ind, cell in enumerate(cells):
    data_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['NaCl']:
                for iteration, trial in enumerate(times):
                    # Index to analysis window
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    data_holder = pd.concat([data_holder, signal], axis=1)
    N_dict[cell] = data_holder
    N_zscore_dict[cell] = data_holder


for cell, cell_df in N_dict.items():
    N_dict[cell] = cell_df.T

# %% Citric Acid

for ind, cell in enumerate(cells):
    data_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['Citric']:
                for iteration, trial in enumerate(times):
                    # Index to analysis window
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]

                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    data_holder = pd.concat([data_holder, signal], axis=1)

    CA_dict[cell] = data_holder
    CA_zscore_dict[cell] = zscore_holder

for cell, cell_df in CA_dict.items():
    CA_dict[cell] = cell_df.T


# %% Quinine

for ind, cell in enumerate(cells):
    data_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['Quinine']:
                for iteration, trial in enumerate(times):
                    # Index to analysis window
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    print(len(data_ind))
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    data_holder = pd.concat([data_holder, signal], axis=1)
    Q_dict[cell] = data_holder

for cell, cell_df in Q_dict.items():
    Q_dict[cell] = cell_df.T

# %% MSG

for ind, cell in enumerate(cells):
    data_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['MSG']:
                for iteration, trial in enumerate(times):
                    # Index to analysis window
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    data_holder = pd.concat([data_holder, signal], axis=1)
    MSG_dict[cell] = data_holder

for cell, cell_df in MSG_dict.items():
    MSG_dict[cell] = cell_df.T

# %% Labels

labt = [0, 5, 34]
labn = [-1, 0, 3]


#%%
as_hm = Plot(data=alldata[0].signals, title='AS', tastant='AS')
as_hm.get_heatmap(AS_dict, tastant='AS', cmap='inferno')


#%%
# suc_hm = Plot(data=alldata[0].signals, title='S', tastant='S')
# suc_hm.get_heatmap(S_dict, tastant='suc', cmap='inferno')

#%%
# nacl_hm = Plot(data=alldata[0].signals, title='NaCl', tastant='NaCl')
# nacl_hm.get_heatmap(N_dict, tastant='NaCl', cmap='inferno')
#%%
# citric_hm = Plot(data=alldata[0].signals, title='Citric', tastant='Citric')
# citric_hm.get_heatmap(CA_dict, tastant='Citric', cmap='inferno')
#%%
# quinine_hm = Plot(data=alldata[0].signals, title='Q', tastant='Q')
# quinine_hm.get_heatmap(Q_dict, tastant='Q', cmap='inferno')
# #%%
msg_hm = Plot(data=alldata[0].signals, title='MSG', tastant='MSG')
msg_hm.get_heatmap(MSG_dict, tastant='MSG', cmap='inferno')


#%%
#
# gradients = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#                       'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#                       'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
#                       'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
#                       'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
# perceptual = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

# for cmap in perceptual:
#     suc_hm = Plot(data=alldata[0].signals, cmap=cmap, title='Sucrose', tastant='Sucrose', )
#     suc_hm.get_heatmap(S_dict, cmap, cmap=cmap)


# %% MSG


#%%
# for cell, df in MSG_dict.items():
#     df_smooth = gaussian_filter(df, sigma=2)
#     b = np.argsort(np.argsort(df_smooth, axis=1), axis=1)
#     im = plt.imshow(b, aspect="auto", cmap="Reds_r")
#
#     plt.title(f'S, {cell}', fontweight='bold')
#     plt.savefig(f'C://Users//dilorenzo//Desktop//CalciumPlots//{cell}, S',
#                 dpi=600,
#                 bbox_inches='tight')
#     plt.show()
#
if __name__ == "__main__":
    pass
