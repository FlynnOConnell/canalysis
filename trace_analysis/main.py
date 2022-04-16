# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import importlib
from copy import copy

import pandas as pd
import logging
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
from pandas import Series

import draw_plots
from core.data import CalciumData
from core.draw_plots import Plot
from scipy.ndimage.filters import gaussian_filter

importlib.reload(draw_plots)


def getPeakWindow(time, peak_time):
    """ Returns the index of tracedata centered 1s around the peak flourescent value for
        that trial.
    """
    aux, window_ind = [], []
    for valor in time:
        aux.append(abs(peak_time - valor))

    window_ind.append(aux.index(min(aux)) - 20)
    window_ind.append(aux.index(min(aux)) + 20)
    return window_ind


def getMatchedTime(time, *argv):
    """  Finds the closest number in tracedata time to the input.
    """
    return_index = []
    for arg in argv:
        temp = []
        for valor in time:
            temp.append(abs(arg - valor))
        return_index.append(temp.index(min(temp)))
    return return_index


# %%
pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')
datadir = 'A:\\'

animal_id = 'PGT13'
# animal_id = 'PGT08'
date = '121421'
alldata = []

data = CalciumData(animal_id, date, datadir, pick=0)
cells = alldata[0].cells

# %%
tracedata = data.tracedata
tracedata.astype(float)
trial_times = data.trial_times
time = data.time
events = [stim for stim in trial_times.keys()]

# %%

count = 0
for cell in tracedata.columns[1:]:
    count += 1

    # HOLDS STIM VALUES FOR EACH CELL
    stim_holder_s = pd.DataFrame(columns=events)
    stim_holder_ss = pd.DataFrame(columns=events)
    stim_holder_w = pd.DataFrame(columns=events)
    stim_holder_sw = pd.DataFrame(columns=events)

    for stim, times in trial_times.items():

        # HOLDS VALUES FOR EACH STIMULUS
        signal_holder = pd.Series(dtype=float)
        window_holder = pd.Series(dtype=float)
        sig_window_holder = pd.Series(dtype=float)
        sig_signal_holder = pd.Series(dtype=float)
        # logging.info(f'{window_holder.name}')

        for iteration, trial in enumerate(times):

            # PULL DATA
            data_ind = np.where(
                (time >= trial) & (time < trial + 4))[0]
            BLtime_ind = np.where(
                (time > trial - 2) & (time <= trial))[0]
            BLtime = time[BLtime_ind]
            bltime = [min(BLtime), max(BLtime)]
            BL_signal = np.array(tracedata.iloc[BLtime_ind, count])
            mean_bl = np.mean(BL_signal)
            stdev_bl = np.std(BL_signal)
            c_interv = mean_bl + stdev_bl * 2.58

            ## SIGNAL ##
            signal = pd.Series(tracedata.iloc[data_ind, count])
            signal.reset_index(drop=True, inplace=True)
            signal_time = np.array(tracedata.iloc[data_ind, 0])
            signal_holder = pd.concat([signal_holder, signal], axis=0, ignore_index=True)

            peak_signal = max(signal)
            peak_ts = tracedata.loc[tracedata[cell]
                                    == peak_signal, 'Time(s)'].iloc[0]

            if peak_ts <= trial + 2.4:
                peak_ts = max(BLtime) + 2.4
                data_ind = np.where(
                    time == peak_ts)[0]
                this_time = time[data_ind]

            # RESPONSE WINDOW
            peak_window_ind = getPeakWindow(time, peak_ts)
            time_lower = time[peak_window_ind[0]]
            time_upper = time[peak_window_ind[1]]
            response_window = pd.Series(tracedata.loc[
                                        peak_window_ind[0]:peak_window_ind[1], cell], name=stim)
            window_holder = pd.concat([window_holder, response_window], axis=0, ignore_index=True)
            window_holder.reset_index(drop=True, inplace=True)

            # SIGNIFICANT TRIALS
            mean_mag = np.mean(response_window)
            dFF = ((mean_mag - mean_bl) / mean_bl)
            # if DF/F is significant, output data
            if dFF >= c_interv:
                sig_signal_holder = pd.concat([sig_signal_holder, signal], axis=0, ignore_index=True)
                sig_window_holder = pd.concat([sig_window_holder, response_window], axis=0, ignore_index=True)

        stim_holder_s[stim] = signal_holder
        stim_holder_ss[stim] = sig_signal_holder
        stim_holder_w[stim] = window_holder
        stim_holder_sw[stim] = sig_window_holder

    signal_dict[cell] = stim_holder_s
    sig_signal_dict[cell] = stim_holder_ss
    window_dict[cell] = stim_holder_w
    sig_window_dict[cell] = stim_holder_sw


if __name__ == "__main__":
    pass
