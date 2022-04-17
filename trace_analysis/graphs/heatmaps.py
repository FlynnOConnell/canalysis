from __future__ import annotations
from typing import Tuple, Iterable, Optional, Sized, Any, Mapping

import importlib
import pandas as pd
import numpy as np
import logging
import scipy.stats as stats
from core.data import CalciumData
from graphs.draw_plots import Plot
from utils import funcs as func
from utils import stats_helpers as stat_help

#%% Data

pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')

datadir = '/Users/flynnoconnell/Documents/Work/Data'


animal_id = 'PGT13'
animal_id = 'PGT08'
date = '070121'


# dates = ['011222', '011422', '011922', '030322', '032422', '120221', '121021', '121721']
# dates = ['060221', '062321', '071621', '072721', '082021', '092321', '093021']
dates = ['011222', '011422', '011922', '030322', '032422', '120221', '121021', '121721']
data = CalciumData(animal_id, date, datadir, pick=0)

alldata = []
for date in dates:
    data_day = CalciumData(animal_id, date, datadir, pick=0)
    print((data_day.session, data_day.cells, data_day.numlicks))
    alldata.append(data_day)

cells = data.cells

# cells = data.date

labt = [0, 5, 34]
labn = [-1, 0, 3]

#%% Initialize Dicts

AS_base_dict, S_base_dict, N_base_dict, CA_base_dict, Q_base_dict, MSG_base_dict = {}, {}, {}, {}, {}, {}
AS_norm, S_norm, N_norm, CA_norm, Q_norm, MSG_norm = {}, {}, {}, {}, {}, {}
AS_window_dict, S_window_dict, N_window_dict, CA_window_dict, Q_window_dict, MSG_window_dict = {}, {}, {}, {}, {}, {}
AS_zscore_dict, S_zscore_dict, N_zscore_dict, CA_zscore_dict, Q_zscore_dict, MSG_zscore_dict = {}, {}, {}, {}, {}, {}
AS_sig_dict, S_sig_dict, N_sig_dict, CA_sig_dict, Q_sig_dict, MSG_sig_dict = {}, {}, {}, {}, {}, {}
AS_sw_dict, S_sw_dict, N_sw_dict, CA_sw_dict, Q_sw_dict, MSG_sw_dict = {}, {}, {}, {}, {}, {}
AS_sz_dict, S_sz_dict, N_sz_dict, CA_sz_dict, Q_sz_dict, MSG_sz_dict = {}, {}, {}, {}, {}, {}

#%% Fill Data Containers

aaa_cellpd = pd.DataFrame(columns=cells)
aab_cellpd = pd.DataFrame(columns=cells)

for cell, df in AS_base_dict.items(): 
    df = df.T
    cell_holder = []   
    for trial, ser in df.items(): 
        cell_holder.extend(ser)
    aaa_cellpd[cell] = cell_holder
        
for cell, df in AS_base_dict.items(): 
    df = df.T
    cell_holder = func.df_tolist(df)
    aab_cellpd[cell] = cell_holder
    
    # for col in df.columns:
    #     print(df[col])
        
    #     this_ser = df[col]
    #     this_ser = this_ser.tolist()
    #     cell_holder.append(this_ser)
    # cellpd[cell] = cell_holder

tastant = ['ArtSal', 'MSG', 'Quinine', 'NaCl', 'Citric']
a, aa, aaa, aaaa, aaaaa, aaaaaaa, aaaaaaaa = stat_help.get_tastant_dicts(tastant, cells, data)
#%%

    ###### ArtSal ########
for ind, cell in enumerate(cells):
    signal_holder = pd.DataFrame()
    window_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()
    norm_holder = pd.DataFrame()
    sig_signal_holder = pd.DataFrame()
    sig_window_holder = pd.DataFrame()
    sig_zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['ArtSal']:
                for iteration, trial in enumerate(times):
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    BLtime_ind = np.where(
                        (date.time > trial - 2) & (date.time < trial))[0]
                    BLtime = date.time[BLtime_ind]
                    bltime = [min(BLtime), max(BLtime)]
                    BL_signal = np.array(date.tracedata.iloc[BLtime_ind, ind])
                    mean_bl = np.mean(BL_signal)
                    stdev_bl = np.std(BL_signal)
                    c_interv = mean_bl + stdev_bl * 2.58
                    ## SIGNAL
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    peak_signal = max(signal)
                    peak_ts = date.tracedata.loc[date.tracedata[cell]
                                                 == peak_signal, 'Time(s)'].iloc[0]
                    if peak_ts <= trial + 2.4:
                        peak_ts = max(BLtime) + 2.4
                        data_ind = np.where(
                            date.time == peak_ts)[0]
                        this_time = date.time[data_ind]
                    peak_window_ind = func.get_peak_window(date.time, peak_ts)
                    time_lower = date.time[peak_window_ind[0]]
                    time_upper = date.time[peak_window_ind[1]]
                    response_window = pd.Series(date.tracedata.loc[
                                                peak_window_ind[0]:peak_window_ind[1], cell], name=stim)
                    response_window.reset_index(drop=True, inplace=True)
                    window_holder = pd.concat([window_holder, response_window], axis=1)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    signal_holder = pd.concat([signal_holder, signal], axis=1)

                    norm = (signal - np.mean(signal[:7])) / np.mean(signal[:7])
                    norm_holder = pd.concat([norm_holder, norm], axis=1)

                    # SIGNIFICANT TRIALS
                    mean_mag = np.mean(response_window)
                    dFF = ((mean_mag - mean_bl) / mean_bl)
                    # if DF/F is significant, output data
                    if dFF >= c_interv:
                        sig_signal_holder = pd.concat([sig_signal_holder, signal], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, response_window], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, zscore], axis=1)
                    else:
                        no_resp_s = pd.Series(index=np.arange(len(signal)), dtype=int)
                        no_resp_rw = pd.Series(index=np.arange(len(response_window)), dtype=int)
                        no_resp_z = pd.Series(index=np.arange(len(zscore)), dtype=int)
                        sig_signal_holder = pd.concat([sig_signal_holder, no_resp_s], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, no_resp_rw], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, no_resp_z], axis=1)

    AS_norm[cell] = norm_holder
    AS_base_dict[cell] = signal_holder
    AS_zscore_dict[cell] = zscore_holder
    AS_sw_dict[cell] = sig_window_holder
    AS_sz_dict[cell] = sig_zscore_holder
    AS_window_dict[cell] = window_holder
    AS_sig_dict[cell] = sig_signal_holder
    
    cell_holder = []   
    for trial, ser in norm_holder.items(): 
        cell_holder.extend(ser)
    aaa_cellpd[cell] = cell_holder    

for cell, cell_df in AS_base_dict.items():
    AS_base_dict[cell] = cell_df.T
for cell, cell_df in AS_zscore_dict.items():
    AS_zscore_dict[cell] = cell_df.T
for cell, cell_df in AS_sw_dict.items():
    AS_sw_dict[cell] = cell_df.T
for cell, cell_df in AS_sz_dict.items():
    AS_sz_dict[cell] = cell_df.T
for cell, cell_df in AS_window_dict.items():
    AS_window_dict[cell] = cell_df.T
for cell, cell_df in AS_sig_dict.items():
    AS_sig_dict[cell] = cell_df.T


#%%
     ###### SUCROSE ########
for ind, cell in enumerate(cells):
    signal_holder = pd.DataFrame()
    window_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()
    norm_holder = pd.DataFrame()
    sig_signal_holder = pd.DataFrame()
    sig_window_holder = pd.DataFrame()
    sig_zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['Sucrose']:
                for iteration, trial in enumerate(times):
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    BLtime_ind = np.where(
                        (date.time > trial - 2) & (date.time < trial))[0]
                    BLtime = date.time[BLtime_ind]
                    bltime = [min(BLtime), max(BLtime)]
                    BL_signal = np.array(date.tracedata.iloc[BLtime_ind, ind])
                    mean_bl = np.mean(BL_signal)
                    stdev_bl = np.std(BL_signal)
                    c_interv = mean_bl + stdev_bl * 2.58
                    ## SIGNAL
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    peak_signal = max(signal)
                    peak_ts = date.tracedata.loc[date.tracedata[cell]
                                            == peak_signal, 'Time(s)'].iloc[0]
                    if peak_ts <= trial + 2.4:
                        peak_ts = max(BLtime) + 2.4
                        data_ind = np.where(
                            date.time == peak_ts)[0]
                        this_time = date.time[data_ind]
                    peak_window_ind = func.get_peak_window(date.time, peak_ts)
                    time_lower = date.time[peak_window_ind[0]]
                    time_upper = date.time[peak_window_ind[1]]
                    response_window = pd.Series(date.tracedata.loc[
                                                peak_window_ind[0]:peak_window_ind[1], cell], name=stim)
                    response_window.reset_index(drop=True, inplace=True)
                    window_holder = pd.concat([window_holder, response_window], axis=1)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    signal_holder = pd.concat([signal_holder, signal], axis=1)

                    norm = (signal - np.mean(signal[:7]))/np.mean(signal[:7])
                    norm_holder = pd.concat([norm_holder, norm], axis=1)

                    # SIGNIFICANT TRIALS
                    mean_mag = np.mean(response_window)
                    dFF = ((mean_mag - mean_bl) / mean_bl)
                    # if DF/F is significant, output data
                    if dFF >= c_interv:
                        sig_signal_holder = pd.concat([sig_signal_holder, signal], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, response_window], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, zscore], axis=1)
                    else:
                        no_resp_s = pd.Series(index=np.arange(len(signal)), dtype=int)
                        no_resp_rw = pd.Series(index=np.arange(len(response_window)), dtype=int)
                        no_resp_z = pd.Series(index=np.arange(len(zscore)), dtype=int)
                        sig_signal_holder = pd.concat([sig_signal_holder, no_resp_s], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, no_resp_rw], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, no_resp_z], axis=1)

    S_norm[cell] = norm_holder
    S_base_dict[cell] = signal_holder
    S_zscore_dict[cell] = zscore_holder
    S_sw_dict[cell] = sig_window_holder
    S_sz_dict[cell] = sig_zscore_holder
    S_window_dict[cell] = window_holder
    S_sig_dict[cell] = sig_signal_holder

for cell, cell_df in S_base_dict.items():
    S_base_dict[cell] = cell_df.T
for cell, cell_df in S_zscore_dict.items():
    S_zscore_dict[cell] = cell_df.T
for cell, cell_df in S_sw_dict.items():
    S_sw_dict[cell] = cell_df.T
for cell, cell_df in S_sz_dict.items():
    S_sz_dict[cell] = cell_df.T
for cell, cell_df in S_window_dict.items():
    S_window_dict[cell] = cell_df.T
for cell, cell_df in S_sig_dict.items():
    S_sig_dict[cell] = cell_df.T
for cell, cell_df in S_norm.items():
    S_norm[cell] = cell_df.T

     ####### NaCl #########
for ind, cell in enumerate(cells):
    signal_holder = pd.DataFrame()
    window_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()
    norm_holder = pd.DataFrame()
    sig_signal_holder = pd.DataFrame()
    sig_window_holder = pd.DataFrame()
    sig_zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['NaCl']:
                for iteration, trial in enumerate(times):
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    BLtime_ind = np.where(
                        (date.time > trial - 2) & (date.time < trial))[0]
                    BLtime = date.time[BLtime_ind]
                    bltime = [min(BLtime), max(BLtime)]
                    BL_signal = np.array(date.tracedata.iloc[BLtime_ind, ind])
                    mean_bl = np.mean(BL_signal)
                    stdev_bl = np.std(BL_signal)
                    c_interv = mean_bl + stdev_bl * 2.58
                    ## SIGNAL
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    peak_signal = max(signal)
                    peak_ts = date.tracedata.loc[date.tracedata[cell]
                                            == peak_signal, 'Time(s)'].iloc[0]
                    if peak_ts <= trial + 2.4:
                        peak_ts = max(BLtime) + 2.4
                        data_ind = np.where(
                            date.time == peak_ts)[0]
                        this_time = date.time[data_ind]
                    peak_window_ind = func.get_peak_window(date.time, peak_ts)
                    time_lower = date.time[peak_window_ind[0]]
                    time_upper = date.time[peak_window_ind[1]]
                    response_window = pd.Series(date.tracedata.loc[
                                                peak_window_ind[0]:peak_window_ind[1], cell], name=stim)
                    response_window.reset_index(drop=True, inplace=True)
                    window_holder = pd.concat([window_holder, response_window], axis=1)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    signal_holder = pd.concat([signal_holder, signal], axis=1)

                    norm = (signal - np.mean(signal[:7]))/np.mean(signal[:7])
                    norm_holder = pd.concat([norm_holder, norm], axis=1)

                    # SIGNIFICANT TRIALS
                    mean_mag = np.mean(response_window)
                    dFF = ((mean_mag - mean_bl) / mean_bl)
                    # if DF/F is significant, output data
                    if dFF >= c_interv:
                        sig_signal_holder = pd.concat([sig_signal_holder, signal], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, response_window], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, zscore], axis=1)
                    else:
                        no_resp_s = pd.Series(index=np.arange(len(signal)), dtype=int)
                        no_resp_rw = pd.Series(index=np.arange(len(response_window)), dtype=int)
                        no_resp_z = pd.Series(index=np.arange(len(zscore)), dtype=int)
                        sig_signal_holder = pd.concat([sig_signal_holder, no_resp_s], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, no_resp_rw], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, no_resp_z], axis=1)

    N_norm[cell] = norm_holder
    N_base_dict[cell] = signal_holder
    N_zscore_dict[cell] = zscore_holder
    N_sw_dict[cell] = sig_window_holder
    N_sz_dict[cell] = sig_zscore_holder
    N_window_dict[cell] = window_holder
    N_sig_dict[cell] = sig_signal_holder

for cell, cell_df in N_base_dict.items():
    N_base_dict[cell] = cell_df.T
for cell, cell_df in N_zscore_dict.items():
    N_zscore_dict[cell] = cell_df.T
for cell, cell_df in N_sw_dict.items():
    N_sw_dict[cell] = cell_df.T
for cell, cell_df in N_sz_dict.items():
    N_sz_dict[cell] = cell_df.T
for cell, cell_df in N_window_dict.items():
    N_window_dict[cell] = cell_df.T
for cell, cell_df in N_sig_dict.items():
    N_sig_dict[cell] = cell_df.T
for cell, cell_df in N_norm.items():
    N_norm[cell] = cell_df.T

    ###### Citric Acid ########
for ind, cell in enumerate(cells):
    signal_holder = pd.DataFrame()
    window_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()
    norm_holder = pd.DataFrame()
    sig_signal_holder = pd.DataFrame()
    sig_window_holder = pd.DataFrame()
    sig_zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['Citric']:
                for iteration, trial in enumerate(times):
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    BLtime_ind = np.where(
                        (date.time > trial - 2) & (date.time < trial))[0]
                    BLtime = date.time[BLtime_ind]
                    bltime = [min(BLtime), max(BLtime)]
                    BL_signal = np.array(date.tracedata.iloc[BLtime_ind, ind])
                    mean_bl = np.mean(BL_signal)
                    stdev_bl = np.std(BL_signal)
                    c_interv = mean_bl + stdev_bl * 2.58
                    ## SIGNAL
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    peak_signal = max(signal)
                    peak_ts = date.tracedata.loc[date.tracedata[cell]
                                            == peak_signal, 'Time(s)'].iloc[0]
                    if peak_ts <= trial + 2.4:
                        peak_ts = max(BLtime) + 2.4
                        data_ind = np.where(
                            date.time == peak_ts)[0]
                        this_time = date.time[data_ind]
                    peak_window_ind = func.get_peak_window(date.time, peak_ts)
                    time_lower = date.time[peak_window_ind[0]]
                    time_upper = date.time[peak_window_ind[1]]
                    response_window = pd.Series(date.tracedata.loc[
                                                peak_window_ind[0]:peak_window_ind[1], cell], name=stim)
                    response_window.reset_index(drop=True, inplace=True)
                    window_holder = pd.concat([window_holder, response_window], axis=1)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    signal_holder = pd.concat([signal_holder, signal], axis=1)

                    norm = (signal - np.mean(signal[:7]))/np.mean(signal[:7])
                    norm_holder = pd.concat([norm_holder, norm], axis=1)

                    # SIGNIFICANT TRIALS
                    mean_mag = np.mean(response_window)
                    dFF = ((mean_mag - mean_bl) / mean_bl)
                    # if DF/F is significant, output data
                    if dFF >= c_interv:
                        sig_signal_holder = pd.concat([sig_signal_holder, signal], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, response_window], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, zscore], axis=1)
                    else:
                        no_resp_s = pd.Series(index=np.arange(len(signal)), dtype=int)
                        no_resp_rw = pd.Series(index=np.arange(len(response_window)), dtype=int)
                        no_resp_z = pd.Series(index=np.arange(len(zscore)), dtype=int)
                        sig_signal_holder = pd.concat([sig_signal_holder, no_resp_s], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, no_resp_rw], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, no_resp_z], axis=1)

    CA_norm[cell] = norm_holder
    CA_zscore_dict[cell] = zscore_holder
    CA_sw_dict[cell] = sig_window_holder
    CA_sz_dict[cell] = sig_zscore_holder
    CA_window_dict[cell] = window_holder
    CA_sig_dict[cell] = sig_signal_holder

for cell, cell_df in CA_base_dict.items():
    CA_base_dict[cell] = cell_df.T
for cell, cell_df in CA_zscore_dict.items():
    CA_zscore_dict[cell] = cell_df.T
for cell, cell_df in CA_sw_dict.items():
    CA_sw_dict[cell] = cell_df.T
for cell, cell_df in CA_sz_dict.items():
    CA_sz_dict[cell] = cell_df.T
for cell, cell_df in CA_window_dict.items():
    CA_window_dict[cell] = cell_df.T
for cell, cell_df in CA_sig_dict.items():
    CA_sig_dict[cell] = cell_df.T
for cell, cell_df in CA_norm.items():
    CA_norm[cell] = cell_df.T

    ####### Quinine ########
for ind, cell in enumerate(cells):
    signal_holder = pd.DataFrame()
    window_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()
    norm_holder = pd.DataFrame()
    sig_signal_holder = pd.DataFrame()
    sig_window_holder = pd.DataFrame()
    sig_zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['Quinine']:
                for iteration, trial in enumerate(times):
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    BLtime_ind = np.where(
                        (date.time > trial - 2) & (date.time < trial))[0]
                    BLtime = date.time[BLtime_ind]
                    bltime = [min(BLtime), max(BLtime)]
                    BL_signal = np.array(date.tracedata.iloc[BLtime_ind, ind])
                    mean_bl = np.mean(BL_signal)
                    stdev_bl = np.std(BL_signal)
                    c_interv = mean_bl + stdev_bl * 2.58
                    ## SIGNAL
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    peak_signal = max(signal)
                    peak_ts = date.tracedata.loc[date.tracedata[cell]
                                            == peak_signal, 'Time(s)'].iloc[0]
                    if peak_ts <= trial + 2.4:
                        peak_ts = max(BLtime) + 2.4
                        data_ind = np.where(
                            date.time == peak_ts)[0]
                        this_time = date.time[data_ind]
                    peak_window_ind = func.get_peak_window(date.time, peak_ts)
                    time_lower = date.time[peak_window_ind[0]]
                    time_upper = date.time[peak_window_ind[1]]
                    response_window = pd.Series(date.tracedata.loc[
                                                peak_window_ind[0]:peak_window_ind[1], cell], name=stim)
                    response_window.reset_index(drop=True, inplace=True)
                    window_holder = pd.concat([window_holder, response_window], axis=1)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    signal_holder = pd.concat([signal_holder, signal], axis=1)

                    norm = (signal - np.mean(signal[:7]))/np.mean(signal[:7])
                    norm_holder = pd.concat([norm_holder, norm], axis=1)

                    # SIGNIFICANT TRIALS
                    mean_mag = np.mean(response_window)
                    dFF = ((mean_mag - mean_bl) / mean_bl)
                    # if DF/F is significant, output data
                    if dFF >= c_interv:
                        sig_signal_holder = pd.concat([sig_signal_holder, signal], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, response_window], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, zscore], axis=1)
                    else:
                        no_resp_s = pd.Series(index=np.arange(len(signal)), dtype=int)
                        no_resp_rw = pd.Series(index=np.arange(len(response_window)), dtype=int)
                        no_resp_z = pd.Series(index=np.arange(len(zscore)), dtype=int)
                        sig_signal_holder = pd.concat([sig_signal_holder, no_resp_s], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, no_resp_rw], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, no_resp_z], axis=1)

    Q_norm[cell] = norm_holder
    Q_base_dict[cell] = signal_holder
    Q_zscore_dict[cell] = zscore_holder
    Q_sw_dict[cell] = sig_window_holder
    Q_sz_dict[cell] = sig_zscore_holder
    Q_window_dict[cell] = window_holder
    Q_sig_dict[cell] = sig_signal_holder

for cell, cell_df in Q_base_dict.items():
    Q_base_dict[cell] = cell_df.T
for cell, cell_df in Q_zscore_dict.items():
    Q_zscore_dict[cell] = cell_df.T
for cell, cell_df in Q_sw_dict.items():
    Q_sw_dict[cell] = cell_df.T
for cell, cell_df in Q_sz_dict.items():
    Q_sz_dict[cell] = cell_df.T
for cell, cell_df in Q_window_dict.items():
    Q_window_dict[cell] = cell_df.T
for cell, cell_df in Q_sig_dict.items():
    Q_sig_dict[cell] = cell_df.T
for cell, cell_df in S_norm.items():
    S_norm[cell] = cell_df.T

     ####### MSG ########
for ind, cell in enumerate(cells):
    signal_holder = pd.DataFrame()
    window_holder = pd.DataFrame()
    zscore_holder = pd.DataFrame()
    norm_holder = pd.DataFrame()
    sig_signal_holder = pd.DataFrame()
    sig_window_holder = pd.DataFrame()
    sig_zscore_holder = pd.DataFrame()

    for date in alldata:
        for stim, times in date.trial_times.items():
            if stim in ['MSG']:
                for iteration, trial in enumerate(times):
                    data_ind = np.where(
                        (date.time > trial - .5) & (date.time < trial + 3))[0]
                    data_ind = data_ind[:36]
                    BLtime_ind = np.where(
                        (date.time > trial - 2) & (date.time < trial))[0]
                    BLtime = date.time[BLtime_ind]
                    bltime = [min(BLtime), max(BLtime)]
                    BL_signal = np.array(date.tracedata.iloc[BLtime_ind, ind])
                    mean_bl = np.mean(BL_signal)
                    stdev_bl = np.std(BL_signal)
                    c_interv = mean_bl + stdev_bl * 2.58
                    ## SIGNAL
                    signal = (date.signals.iloc[data_ind, ind])
                    signal = signal.astype(float)
                    signal.reset_index(drop=True, inplace=True)
                    signal.rename(date.date, inplace=True)
                    zscore = pd.Series(stats.zscore(signal))
                    zscore.reset_index(drop=True, inplace=True)
                    peak_signal = max(signal)
                    peak_ts = date.tracedata.loc[date.tracedata[cell]
                                            == peak_signal, 'Time(s)'].iloc[0]
                    if peak_ts <= trial + 2.4:
                        peak_ts = max(BLtime) + 2.4
                        data_ind = np.where(
                            date.time == peak_ts)[0]
                        this_time = date.time[data_ind]

                    peak_window_ind = func.get_peak_window(date.time, peak_ts)
                    time_lower = date.time[peak_window_ind[0]]
                    time_upper = date.time[peak_window_ind[1]]
                    response_window = pd.Series(date.tracedata.loc[
                                                peak_window_ind[0]:peak_window_ind[1],
                                                cell],
                                                name=stim)
                    response_window.reset_index(drop=True, inplace=True)
                    window_holder = pd.concat([window_holder, response_window], axis=1)
                    zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                    signal_holder = pd.concat([signal_holder, signal], axis=1)
                    norm = (signal - np.mean(signal[:7]))/np.mean(signal[:7])
                    norm_holder = pd.concat([norm_holder, norm], axis=1)
                    # SIGNIFICANT TRIALS
                    mean_mag = np.mean(response_window)
                    dFF = ((mean_mag - mean_bl) / mean_bl)

                    if dFF >= c_interv:
                        sig_signal_holder = pd.concat([sig_signal_holder, signal], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, response_window], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, zscore], axis=1)
                    else:
                        no_resp_s = pd.Series(index=np.arange(len(signal)), dtype=int)
                        no_resp_rw = pd.Series(index=np.arange(len(response_window)), dtype=int)
                        no_resp_z = pd.Series(index=np.arange(len(zscore)), dtype=int)
                        sig_signal_holder = pd.concat([sig_signal_holder, no_resp_s], axis=1)
                        sig_window_holder = pd.concat([sig_window_holder, no_resp_rw], axis=1)
                        sig_zscore_holder = pd.concat([sig_zscore_holder, no_resp_z], axis=1)


    MSG_norm[cell] = norm_holder
    MSG_base_dict[cell] = signal_holder
    MSG_zscore_dict[cell] = zscore_holder
    MSG_sw_dict[cell] = sig_window_holder
    MSG_sz_dict[cell] = sig_zscore_holder
    MSG_window_dict[cell] = window_holder
    MSG_sig_dict[cell] = sig_signal_holder

for cell, cell_df in MSG_base_dict.items():
    MSG_base_dict[cell] = cell_df.T
for cell, cell_df in MSG_zscore_dict.items():
    MSG_zscore_dict[cell] = cell_df.T
for cell, cell_df in MSG_sw_dict.items():
    MSG_sw_dict[cell] = cell_df.T
for cell, cell_df in MSG_sz_dict.items():
    MSG_sz_dict[cell] = cell_df.T
for cell, cell_df in MSG_window_dict.items():
    MSG_window_dict[cell] = cell_df.T
for cell, cell_df in MSG_sig_dict.items():
    MSG_sig_dict[cell] = cell_df.T
for cell, cell_df in MSG_norm.items():
    MSG_norm[cell] = cell_df.T

#%% PLOTS
as_hm = Plot(data=alldata[0].signals, title='zscore', tastant='AS')
as_hm.get_heatmap(AS_sw_dict, tastant='AS_window', cmap='inferno')

#%%
suc_hm = Plot(data=alldata[0].signals, title='zscore', tastant='S')
suc_hm.get_heatmap(S_sw_dict, tastant='suc_window', cmap='inferno')

#%%
nacl_hm = Plot(data=alldata[0].signals, title='zscore', tastant='NaCl')
nacl_hm.get_heatmap(N_sw_dict, tastant='NaCl_window', cmap='inferno')

if __name__ == "__main__":
    pass

