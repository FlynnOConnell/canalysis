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

def get_hsvcmap(i, N, rot=0.):
    nsc = 24
    chsv = mcolors.rgb_to_hsv(plt.cm.hsv(((np.arange(N)/N)+rot) % 1.)[i,:3])
    rhsv = mcolors.rgb_to_hsv(plt.cm.Reds(np.linspace(.2,1,nsc))[:,:3])
    arhsv = np.tile(chsv,nsc).reshape(nsc,3)
    arhsv[:,1:] = rhsv[:,1:]
    rgb = mcolors.hsv_to_rgb(arhsv)
    return mcolors.LinearSegmentedColormap.from_list("",rgb)


def columnwise_heatmap(array, ax=None, cm='magma', **kw):
    ax = ax or plt.gca()
    ax.axis('off')
    premask = np.tile(np.arange(array.shape[1]), array.shape[0]).reshape(array.shape)
    images = []
    for i in range(array.shape[1]):
        col = np.ma.array(array, mask = premask != i)
        im = sns.heatmap(col, cmap=cm, square=True)
        # im = ax.imshow(col, cmap=cm, **kw)
        images.append(im)
    return images

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


#%% Data

pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')
datadir = '/Users/flynnoconnell/Documents/Work/Data'
animal_id = 'PGT13'
# animal_id = 'PGT08'
# date = '070121'
# dates = ['060221', '062321', '071621', '072721', '082021', '092321', '093021']
dates = ['011222', '011422', '011922', '030322', '032422', '120221', '121021', '121721']
# data = CalciumData(animal_id, date, datadir, pick=0)

alldata = []
for date in dates:
    data_day = CalciumData(animal_id, date, datadir, pick=0)
    print((data_day.session, data_day.cells, data_day.numlicks))
    alldata.append(data_day)
sessions = {}
for cl in alldata:
    sessions[cl.date] = cl
    

cells = alldata[1].cells
# cells = data.date
data = alldata[0].tastants
labt = [0, 5, 34]
labn = [-1, 0, 3]


#%% Fill Data Containers

AS_base_dict, AS_norm, AS_zscore_dict, AS_sw_dict, AS_sz_dict, AS_window_dict, AS_sig_dict, AS_pct_dict = stat_help.get_tastant_dicts('ArtSal', cells, alldata)

S_base_dict, S_norm, S_zscore_dict, S_sw_dict, S_sz_dict, S_window_dict, S_sig_dict, S_pct_dict = stat_help.get_tastant_dicts('Sucrose', cells, alldata)

N_base_dict, N_norm, N_zscore_dict, N_sw_dict, N_sz_dict, N_window_dict, N_sig_dict, N_pct_dict = stat_help.get_tastant_dicts('NaCl', cells, alldata)

CA_base_dict, CA_norm, CA_zscore_dict, CA_sw_dict, CA_sz_dict, CA_window_dict, CA_sig_dict, CA_pct_dict = stat_help.get_tastant_dicts('Citric', cells, alldata)

Q_base_dict, Q_norm, Q_zscore_dict, Q_sw_dict, Q_sz_dict, Q_window_dict, Q_sig_dict, Q_pct_dict = stat_help.get_tastant_dicts('Quinine', cells, alldata)

MSG_base_dict, MSG_norm, MSG_zscore_dict, MSG_sw_dict, MSG_sz_dict, MSG_window_dict, MSG_sig_dict, MSG_pct_dict = stat_help.get_tastant_dicts('MSG', cells, alldata)

#%%


sns_heatmap(AS_zscore_dict,
            tastant='AS',
            plottype='zscore',
            sigma=2,
            square=True,
            cbar=True,
            y=30,
            linewidth=3,
            color='white',
            robust=True,
            save=True)

sns_heatmap(S_zscore_dict,
            tastant='S',
            plottype='zscore',
            sigma=2,
            square=True,
            cbar=True,
            y=30,
            linewidth=3,
            color='white',
            robust=False,
            save=True)

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





for cell, df in AS_zscore_dict.items():
    tastant = 'AS'
    df = df.T
    df_smooth = pd.DataFrame(gaussian_filter(df, sigma=2))
    cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)
    ### Plot data
    fig, ax = plt.subplots()
    ax = sns.heatmap(df_smooth,square=True, cbar=False)
    ax.axis('off')
    # ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
    ax.axhline(y=30, color='white', linewidth=3)
    plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{cell}_{tastant}_{ty}.png',
                dpi=400, bbox_inches='tight', pad_inches=0.01)

for cell, df in AS_zscore_dict.items():
    tastant = 'AS'

    df = df.T
    df_smooth = pd.DataFrame(gaussian_filter(df, sigma=2.5))
    cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)
    ### Plot data
    fig, ax = plt.subplots()
    ax.axis('off')
    ax = sns.heatmap(df_smooth,square=True, cbar=False)
    ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
    ax.axhline(y=30, color='white', linewidth=3)
    tastant='test'
    ty = 'z'
    plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{cell}_{tastant}_{ty}.png',
                dpi=400, bbox_inches='tight', pad_inches=0.01)
    
for cell, df in N_norm.items():
    df = df.T
    df_smooth = pd.DataFrame(gaussian_filter(df, sigma=2))
    cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)

    ### Plot data
    fig, ax = plt.subplots(figsize=(8,4.5))
    ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
    ax.axhline(y=30, color='white', linewidth=4)
    ty = 'zscore'
    plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{cell}_{tastant}_{ty}.png',
                dpi=400, bbox_inches='tight', pad_inches=0.01)


for cell, df in CA_sz_dict.items():
    df = df.T
    df_smooth = pd.DataFrame(gaussian_filter(df, sigma=2))
    cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)
    ### Plot data
    fig, ax = plt.subplots(figsize=(8,4.5))
    ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
    ax.axhline(y=10, color='white', linewidth=4)
    ty = 'zscore'
    plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{cell}_{tastant}_{ty}.png',
                dpi=400, bbox_inches='tight', pad_inches=0.01)


for cell, df in Q_sz_dict.items():
    df = df.T
    df_smooth = pd.DataFrame(gaussian_filter(df, sigma=2))
    cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)

    ### Plot data
    fig, ax = plt.subplots(figsize=(8,4.5))
    ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
    ax.axhline(y=10, color='white', linewidth=4)
    plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{cell}_{tastant}_{ty}.png', 
                dpi=400, bbox_inches='tight', pad_inches=0.01)
    
for cell, df in MSG_sz_dict.items():
    df = df.T
    df_smooth = pd.DataFrame(gaussian_filter(df, sigma=2))
    cm = sns.color_palette('magma', n_colors = df.shape[1], as_cmap = True)

    ### Plot data
    fig, ax = plt.subplots(figsize=(8,4.5))
    ims = columnwise_heatmap(df_smooth.values, ax=ax, aspect="auto")
    ax.axhline(y=10, color='white', linewidth=4)
    plt.savefig(f'/Users/flynnoconnell/Pictures/heatmaps/{cell}_{tastant}_{ty}.png',
                dpi=400, bbox_inches='tight', pad_inches=0.01)
    
#%% PCA


def something(date, dct):
    holder = []
    for cell, df in dct.items(): 
        holder = []
        df = df.T
        df.dropna(inplace=True)
        
        for col in df.columns:
            holder.extend(df[col])
    return holder

s_pd = pd.DataFrame()
for cell, df in S_base_dict.items(): 
    df = df.T
    df.dropna(inplace=True)
    cell_holder = []
    for trial, ser in df.items():
        cell_holder.extend(ser)
    s_pd[cell] = cell_holder
    s_pd['colors'] = 'magenta'

n_pd = pd.DataFrame()
for cell, df in N_base_dict.items(): 
    df = df.T
    df.dropna(inplace=True)
    cell_holder = []
    for trial, ser in df.items():
        cell_holder.extend(ser)
    n_pd[cell] = cell_holder
    n_pd['colors'] = 'lime'
    
ca_pd = pd.DataFrame()
for cell, df in CA_base_dict.items(): 
    df = df.T
    df.dropna(inplace=True)
    cell_holder = []
    for trial, ser in df.items():
        cell_holder.extend(ser)
    ca_pd[cell] = cell_holder
    ca_pd['colors'] = 'yellow'

q_pd = pd.DataFrame()
for cell, df in Q_base_dict.items(): 
    df = df.T
    df.dropna(inplace=True)
    cell_holder = []
    for trial, ser in df.items():
        cell_holder.extend(ser)
    q_pd[cell] = cell_holder
    q_pd['colors'] = 'red'

msg_pd = pd.DataFrame()
for cell, df in MSG_base_dict.items(): 
    df = df.T
    df.dropna(inplace=True)
    cell_holder = []
    for trial, ser in df.items():
        cell_holder.extend(ser)
    msg_pd[cell] = cell_holder
    msg_pd['colors'] = 'red'


aaa = something('011222', S_base_dict)
data = alldata[0]

all_base_pd = pd.DataFrame()
all_base_pd = pd.concat([as_pd, s_pd, n_pd, ca_pd, msg_pd, q_pd], axis=0, ignore_index=True)
colors = all_base_pd.pop('colors')

all_base_pd.shape
colors.shape

pca_df, var, lab = data.PCA(all_base_pd, colors=colors, numcomp=2)
pca_graph = Plot(pca_df, colors)
pca_graph.scatter_2d(df=pca_df, colors=colors, lim=True)

pca_df.min(numeric_only=True)
pca_df.max(numeric_only=True)

#%%

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

a_df = S_zscore_dict['C02']
a_df.reset_index(drop=True, inplace=True)
 a_df.stack().reset_index(name='value').rename(columns={'level_0':'x', 'level_1': 'y'})
from sklearn.preprocessing import StandardScaler
 
aaa_df = StandardScaler().fit_transform(aa_df)
X = aa_df.x
Y = aa_df.y
Z = aa_df.value

# X,Y = np.meshgrid(X, Y)

for cell, df in N_norm.items():
    df = df.T
    df_smooth = pd.DataFrame(gaussian_filter(df, sigma=2))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_trisurf(X, Y, Z, linewidth=0, antialiased=False, cmap=cm.coolwarm)
# surf = ax.plot_trisurface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# f, axs = plt.subplots(1, df.columns.size, gridspec_kw={'wspace': 0})
# for i, (s, a) in enumerate(zip(df.columns, axs)):
#     sns.heatmap(data=np.array([df[s].values]),annot=False, ax=a, cmap=cm, cbar=False)

#%%

if __name__ == "__main__":
    pass
        
# for cell, df in AS_base_dict.items(): 
#     df = df.T
#     cell_holder = func.df_tolist(df)
#     aab_cellpd[cell] = cell_holder
    

