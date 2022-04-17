#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# stats_helpers.py

Module (core): Functions for pulling stats.

"""
from __future__ import annotations
from typing import Tuple, Iterable, Optional, Any, Mapping

import pandas as pd
import numpy as np
import logging
import scipy.stats as stats
from utils import funcs as func


def get_tastant_dicts(tastant,
              cells,
              alldata: Optional[list | str] = []
              ) -> [pd.DataFrame | dict]:
    
    dct_norm_dict, dct_base_dict, dct_zscore_dict = {}, {}, {} 
    dct_window_dict, dct_sw_dict, dct_sz_dict, dct_sig_dict = {}, {}, {}, {}
    
    for tastant in tastant:
        assert tastant in ['ArtSal', 'MSG', 'Citric', 'Quinine', 'NaCl']
    
    if not isinstance(alldata, Iterable):
        alldata = [alldata]        
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
                if stim in [tastant]:
                    
                    for iteration, trial in enumerate(times):
                        data_ind = np.where(
                            (date.time > trial - .5) & (date.time < trial + 3))[0]
                        data_ind = data_ind[:36]
                        BLtime_ind = np.where(
                            (date.time > trial - 2) & (date.time < trial))[0]
                        BLtime = date.time[BLtime_ind]
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
                        peak_window_ind = func.get_peak_window(date.time, peak_ts)
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
        
        
        dct_base_dict[cell] = signal_holder.T
        dct_norm_dict[cell] = norm_holder.T
        dct_zscore_dict[cell] = zscore_holder.T
        dct_sw_dict[cell] = sig_window_holder.T
        dct_sz_dict[cell] = sig_zscore_holder.T
        dct_window_dict[cell] = window_holder.T
        dct_sig_dict[cell] = sig_signal_holder.T
        
        return dct_base_dict, dct_norm_dict, dct_zscore_dict, dct_sw_dict, dct_sz_dict, dct_window_dict, dct_sig_dict
        
    # for cell, cell_df in dct_base_dict.items():
    #     dct_base_dict[cell] = cell_df.T
    # for cell, cell_df in dct_norm_dict.items():
    #     dct_norm_dict[cell] = cell_df.T
    # for cell, cell_df in dct_zscore_dict.items():
    #     dct_zscore_dict[cell] = cell_df.T
    # for cell, cell_df in dct_sw_dict.items():
    #     dct_sw_dict[cell] = cell_df.T
    # for cell, cell_df in dct_sz_dict.items():
    #     dct_sz_dict[cell] = cell_df.T
    # for cell, cell_df in dct_sig_dict.items():
    #     dct_sig_dict[cell] = cell_df.T
            
        
            

