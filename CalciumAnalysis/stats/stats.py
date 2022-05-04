#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#stats.py

Module(stats): Stats class for PCA and general statistics/output. 
"""
from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from calciumdata import CalciumData
from data_utils import funcs as func
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ProcessData(object):

    def __init__(self, data):
        assert isinstance(data, CalciumData)
        self.tracedata = data.tracedata
        self.time = data.time
        self.cells = data.cells
        self.licktime = data.eventdata.loc[
            data.eventdata['Lick'] == 1, 'time']
        self.get_antibouts()
        self.session = data.session
        self.trial_times = data.trial_times

        self.get_antibouts()
        self.sponts = self.get_sponts()

    @staticmethod
    def PCA(df: pd.DataFrame,
            numcomp: Optional[int] = 2
            ):

        data = StandardScaler().fit_transform(df)
        pca = PCA(n_components=numcomp)
        data_fit = pca.fit_transform(data)

        variance_ns = np.round(
            pca.explained_variance_ratio_ * 100, decimals=1)
        labels = [
            'PC' + str(x) + f' - {variance_ns[x - 1]}%' for x in range(1, len(variance_ns) + 1)]
        df_ns = pd.DataFrame(data_fit, columns=labels)

        return df_ns, variance_ns

    def get_antibouts(self) -> pd.DataFrame | pd.Series:
        antibouts = pd.DataFrame()
        for interv in func.interval(self.licktime, gap=10, outer=True):
            df = self.tracedata.loc[
                (self.tracedata['time'] > (interv[0])) &
                (self.tracedata['time'] < (interv[1]))]
            antibouts = pd.concat([antibouts, df], axis=0)
            antibouts.sort_values(by='time')
        return antibouts

    def get_sponts(self) -> pd.DataFrame | pd.Series:
        sponts = pd.DataFrame()
        for interv in func.interval(self.licktime, gap=30, outer=True):
            df = self.tracedata.loc[
                (self.tracedata['time'] > (interv[0])) &
                (self.tracedata['time'] < (interv[1]))]
            sponts = pd.concat([sponts, df], axis=0)
            sponts.sort_values(by='time')
        return sponts

    def get_stats(self,
                  output: Optional[bool] = False,
                  savedir: Optional[str] = '',
                  ) -> Tuple[pd.DataFrame,
                             pd.DataFrame,
                             pd.DataFrame]:

        firstcell = self.tracedata.columns[1]
        stats_df = pd.DataFrame(columns=[
            'File', 'Cell', 'Stimulus',
            'Trial', 'Baseline (mean)',
            'Baseline (st_dev)',
            'Signal Timestamps (start, stop)',
            'Baseline Timestamps (start, stop)',
            'Shifted?',
            'deltaF/F',
            'Significant?'])
        summary = pd.DataFrame(columns=[
            'File', 'Stimulus', 'Num Trials'])
        raw_df = pd.DataFrame(columns=self.cells)
        count = 0
        stats_list = []
        for cell in self.tracedata.columns[1:]:
            count += 1
            cell_list = []

            for stim, times in self.trial_times.items():
                ntrials = len(times)

                trial_list = []
                for iteration, trial in enumerate(times):

                    # Index to analysis window
                    data_ind = np.where(
                        (self.time > trial) & (self.time < trial + 5))[0]

                    # Index to baseline window (current: 4s pre stim)
                    bltime_ind = np.where(
                        (self.time > trial - 4) & (self.time < trial))[0]
                    bltime = self.time[bltime_ind]
                    bltime = [min(bltime), max(bltime)]

                    # Baseline statistics
                    bl_signal = np.array(self.tracedata.iloc[bltime_ind, count])
                    mean_bl = np.mean(bl_signal)
                    stdev_bl = np.std(bl_signal)
                    c_interv = mean_bl + stdev_bl * 2.58

                    # Get calcium trace & time
                    signal = np.array(self.tracedata.iloc[data_ind, count])

                    # Get peak signal & time
                    peak_signal = max(signal)
                    peak_ts = self.tracedata.loc[self.tracedata[cell]
                                                 == peak_signal, 'time'].iloc[0]

                    if peak_ts <= trial + 2.4:
                        peak_ts = max(bltime) + 2.4
                        shift = 'yes'
                    else:
                        shift = 'no'

                    # Get window 1s centered at peak
                    peak_window_ind = func.get_peak_window(self.time, peak_ts)
                    time_lower = self.time[peak_window_ind[0]]
                    time_upper = self.time[peak_window_ind[1]]
                    response_window = np.array(self.tracedata.loc[
                                               peak_window_ind[0]:peak_window_ind[1], cell])

                    window_ts = [time_lower, time_upper]
                    mean_mag = np.mean(response_window)
                    dff = ((mean_mag - mean_bl) / mean_bl)

                    # if DF/F is significant, output concat new dataframe
                    # with raw values from this trial
                    if dff >= c_interv:

                        d = [stim, 'Trial {}'.format(iteration + 1)]
                        d.extend(
                            ['-' for _ in range(len(self.tracedata.columns[2:]))])

                        track = len(raw_df)

                        sig = 'Significant'
                        raw_df.loc[track] = d

                        baseline_df = (self.tracedata[self.tracedata['time'].between(
                            min(bltime), time_upper)])
                        raw_df = pd.concat([raw_df, baseline_df])

                    else:
                        sig = 'ns'

                    # Check if peak_signal is a duplicate
                    for _ in signal:
                        func.dup_check(signal, peak_signal)

                    # Pack stats into list
                    iteration_list = [self.session, cell, stim,
                                      f'Trial {iteration + 1}',
                                      mean_bl, stdev_bl, window_ts,
                                      bltime, shift, dff, sig]
                    trial_list.append(iteration_list)

                if cell == firstcell:
                    summary_list = [
                        self.session,
                        stim,
                        ntrials
                    ]
                    track = len(summary)
                    summary.loc[track] = summary_list

                cell_list.append(trial_list)
            stats_list.append(cell_list)
        flaten_stats = func.flatten(func.flatten(stats_list))

        # Make dataframe from flattened list of stats
        for ind, d in enumerate(flaten_stats):
            track = len(stats_df)
            stats_df.loc[track] = d
        logging.info('Stats successfully completed.')

        if output:
            print("Outputting data to Excel...")
            with pd.ExcelWriter(str(savedir)
                                + '/'
                                + '_statistics.xlsx') as writer:
                summary.to_excel(
                    writer, sheet_name='Summary', index=False)
                raw_df.to_excel(
                    writer, sheet_name='Raw Data', index=False)
                # lickstats.to_excel(
                #     writer, sheet_name='Lick cells', index=False)
                stats_df.to_excel(
                    writer, sheet_name='Trial Stats', index=False)

            print(' statistics successfully transfered to Excel!')

        return stats_df, raw_df, summary
