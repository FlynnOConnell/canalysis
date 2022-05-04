#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# stats_helpers.py

Module (core): Functions for pulling stats.

"""
from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
import pandas as pd
from calciumdata import CalciumData
from data_utils import funcs as func


def get_tastant_dicts(tastant,
                      cells: Iterable[any],
                      alldata: Optional[list | str | CalciumData] = (),
                      baseline: int | float = 1
                      ) -> [pd.DataFrame | dict]:
    dct_norm_dict, dct_base_dict, dct_zscore_dict, dct_pct_dict = {}, {}, {}, {}
    dct_window_dict, dct_sw_dict, dct_sz_dict, dct_sig_dict = {}, {}, {}, {}

    ###### ArtSal ########
    for ind, cell in enumerate(cells):

        signal_holder = pd.DataFrame()
        window_holder = pd.DataFrame()
        zscore_holder = pd.DataFrame()
        norm_holder = pd.DataFrame()
        pct_holder = pd.DataFrame()
        sig_signal_holder = pd.DataFrame()
        sig_window_holder = pd.DataFrame()
        sig_zscore_holder = pd.DataFrame()

        for date in alldata:
            for stim, times in date.trial_times.items():
                if stim in [tastant]:

                    for iteration, trial in enumerate(times):
                        data_ind = np.where(
                            (date.time > trial - baseline) & (date.time < trial + 4))[0]
                        # data_ind = data_ind[:50]
                        BLtime_ind = np.where(
                            (date.time > trial - baseline) & (date.time < trial))[0]
                        BLtime = date.time[BLtime_ind]
                        BL_signal = np.array(date.tracedata.iloc[BLtime_ind, ind])
                        trial_ind = np.where(
                            (date.time == trial))[0]
                        trial_signal = np.array(date.tracedata.iloc[trial_ind, ind])
                        mean_bl = np.mean(BL_signal)
                        stdev_bl = np.std(BL_signal)
                        c_interv = mean_bl + stdev_bl * 2.58

                        ## SIGNAL
                        signal = (date.signals.iloc[data_ind, ind])
                        signal = signal.astype(float)
                        signal.reset_index(drop=True, inplace=True)
                        signal.rename(date.date, inplace=True)

                        zscore = pd.Series(date.zscores.iloc[data_ind, ind])
                        # zscore = pd.Series(stats.zscore(signal))
                        zscore.reset_index(drop=True, inplace=True)

                        peak_signal = max(signal)
                        peak_ts = date.tracedata.loc[date.tracedata[cell]
                                                     == peak_signal, 'time'].iloc[0]
                        if peak_ts <= trial + 2.4:
                            peak_ts = max(BLtime) + 2.4
                        peak_window_ind = func.get_peak_window(date.time, peak_ts)
                        response_window = pd.Series(date.tracedata.loc[
                                                    peak_window_ind[0]:peak_window_ind[1], cell], name=stim)
                        response_window.reset_index(drop=True, inplace=True)

                        window_holder = pd.concat([window_holder, response_window], axis=1)
                        zscore_holder = pd.concat([zscore_holder, zscore], axis=1)
                        signal_holder = pd.concat([signal_holder, signal], axis=1)

                        norm = (signal - np.mean(signal[:10])) / np.mean(signal[:10])
                        norm_holder = pd.concat([norm_holder, norm], axis=1)

                        # SIGNIFICANT TRIALS
                        mean_mag = np.mean(signal[10:40])
                        dFF = ((mean_mag - mean_bl) / mean_bl)

                        tmp = []
                        for value in signal:
                            val = (value - trial_signal) / trial_signal
                            tmp.append(val)
                        pct_change = pd.Series(data=tmp, dtype=float)
                        pct_holder = pd.concat([pct_holder, pct_change], axis=1)

                        # if DF/F is significant, output data
                        if dFF >= c_interv:
                            sig_signal_holder = pd.concat([sig_signal_holder, signal], axis=1)
                            sig_window_holder = pd.concat([sig_window_holder, response_window], axis=1)
                            sig_zscore_holder = pd.concat([sig_zscore_holder, zscore], axis=1)
                        else:
                            no_resp_s = pd.Series(index=np.arange(len(signal)), dtype=int).fillna(0)
                            no_resp_rw = pd.Series(index=np.arange(len(response_window)), dtype=int).fillna(0)
                            no_resp_z = pd.Series(index=np.arange(len(zscore)), dtype=int).fillna(0)
                            sig_signal_holder = pd.concat([sig_signal_holder, no_resp_s], axis=1)
                            sig_window_holder = pd.concat([sig_window_holder, no_resp_rw], axis=1)
                            sig_zscore_holder = pd.concat([sig_zscore_holder, no_resp_z], axis=1)

        dct_pct_dict[cell] = signal_holder.T
        dct_base_dict[cell] = signal_holder.T
        dct_norm_dict[cell] = norm_holder.T
        dct_zscore_dict[cell] = zscore_holder.T
        dct_sw_dict[cell] = sig_window_holder.T
        dct_sz_dict[cell] = sig_zscore_holder.T
        dct_window_dict[cell] = window_holder.T
        dct_sig_dict[cell] = sig_signal_holder.T

    return dct_base_dict, dct_norm_dict, \
        dct_zscore_dict, dct_sw_dict, dct_sz_dict, dct_window_dict, dct_sig_dict, dct_pct_dict


def get_single_tastant_dicts(tastant: str,
                             cells,
                             alldata: Optional[tuple | str | list[CalciumData]] = (),
                             baseline: int | float = 1,
                             do_signal=False,
                             do_window=False,
                             do_zscore=False,
                             do_norm=False
                             ) -> [pd.DataFrame | dict]:
    dct_x = {}

    ###### ArtSal ########
    for ind, cell in enumerate(cells):

        pd_x = pd.DataFrame()

        for date in alldata:
            for stim, times in date.trial_times.items():
                if stim in [tastant]:

                    for iteration, trial in enumerate(times):
                        data_ind = np.where(
                            (date.time > trial - baseline) & (date.time < trial + 4))[0]
                        # data_ind = data_ind[:50]
                        BLtime_ind = np.where(
                            (date.time > trial - baseline) & (date.time < trial))[0]
                        BLtime = date.time[BLtime_ind]

                        ## SIGNAL
                        signal = (date.signals.iloc[data_ind, ind])
                        signal = signal.astype(float)
                        signal.reset_index(drop=True, inplace=True)
                        signal.rename(date.date, inplace=True)
                        zscore = pd.Series(date.zscores.iloc[data_ind, ind])
                        # zscore = pd.Series(stats.zscore(signal))
                        zscore.reset_index(drop=True, inplace=True)

                        peak_signal = max(signal)
                        peak_ts = date.tracedata.loc[date.tracedata[cell]
                                                     == peak_signal, 'time'].iloc[0]

                        if peak_ts <= trial + 2.4:
                            peak_ts = max(BLtime) + 2.4

                        peak_window_ind = func.get_peak_window(date.time, peak_ts)
                        response_window = pd.Series(date.tracedata.loc[
                                                    peak_window_ind[0]:peak_window_ind[1], cell], name=stim)
                        response_window.reset_index(drop=True, inplace=True)

                        if do_window:
                            pd_x = pd.concat([pd_x, response_window], axis=1)
                        if do_zscore:
                            pd_x = pd.concat([pd_x, zscore], axis=1)
                        if do_signal:
                            pd_x = pd.concat([pd_x, signal], axis=1)
                        if do_norm:
                            norm = (signal - np.mean(signal[:10])) / np.mean(signal[:10])
                            pd_x = pd.concat([pd_x, norm], axis=1)

        dct_x[cell] = pd_x.T

    return dct_x
