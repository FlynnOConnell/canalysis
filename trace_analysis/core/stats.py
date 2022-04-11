#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:58:23 2022

@author: flynnoconnell
"""
from __future__ import annotations
from typing import Tuple, Iterable, Optional

import pandas as pd
import numpy as np
import logging
from utils import funcs as func


def get_antibouts(data, bouts_td_time: list) -> Tuple[list, list]:
    antibouts_dff = []
    antibouts_td_time = []
    k: int = 0
    cell: str

    for _ in data.tracedata.columns[1:]:
        k += 1
        cell_antibouts = []
        for i in range(len(bouts_td_time) - 1):
            nxt = i + 1

            this_bout = bouts_td_time[i]
            next_bout = bouts_td_time[nxt]
            start = this_bout[1]
            end = next_bout[0]
            # We want anti intervals longer than 10s
            if (end - start) > 10:
                # Get neuron spikes during anti-lick bout
                spikes_ind = list(
                    (np.where((data.tracedata.traces.time > start) & (data.tracedata.traces.time < end)))[0])
                cell_antibouts.append(np.mean(data.tracedata.traces.iloc[spikes_ind, k]))
                if k == 1:
                    antibouts_td_time.append(data.tracedata.traces.time[spikes_ind])

        antibouts_dff.append(np.mean(cell_antibouts))

    return antibouts_dff, antibouts_td_time


def get_sponts(data) -> Tuple[list, list]:
    idxs = np.where(np.diff(data.eventdata.licktime) > 30)[0]

    spont_intervs = np.column_stack((idxs[::2], idxs[1::2]))

    sponts, stdevs = [], []
    for cell in data.tracedata.traces.columns[1:]:
        cell_spont = []
        cell_sd = []
        for bound_low, bound_high in spont_intervs:
            ind0 = func.get_matched_time(data.tracedata.traces.time, bound_low, return_index=True)
            ind1 = func.get_matched_time(data.tracedata.traces.time, bound_high, return_index=True)
            cell_spont.append(
                np.mean(np.array(data.tracedata.traces.loc[ind0[0]:ind1[0], cell])))
            cell_sd.append(
                np.std(np.array(data.tracedata.traces.loc[ind0[0]:ind1[0], cell])))
        mean_spont = np.mean(cell_spont)
        sponts.append(mean_spont)
        mean_stdev = np.mean(cell_sd)
        stdevs.append(mean_stdev)

    return sponts, stdevs


def get_bouts(session, licktime: Iterable, _data) -> pd.DataFrame:
    # Lick Bout
    tmp, bout_intervs, boutstart, boutend, start_traces, end_traces = [], [], [], [], [], []
    for _ in enumerate(licktime):
        bout_intervs = func.interval(licktime)
    if tmp and len(tmp) >= 3:
        bout_intervs.append([tmp[0], tmp[-1]])

    for x in bout_intervs:
        boutstart.append(x[0])
        boutend.append(x[1])

    # Convert to data.tracedata.traces times
    start_traces = func.get_matched_time(_data.tracedata.traces.time, *boutstart)
    end_traces = func.get_matched_time(_data.tracedata.traces.time, *boutend)

    bouts_td_time = list(zip(start_traces, end_traces))

    bout_dff = []
    for i in range(1, len(_data.tracedata.cells) + 1):
        cell_traces = []
        for index, bout in enumerate(bouts_td_time):
            start = bout[0]
            end = bout[1]
            this_bout = (np.where((_data.tracedata.traces.time > start)
                                  & (_data.tracedata.traces.time < end))[0])
            bout_signal = np.mean((_data.tracedata.traces.iloc[this_bout, i]))
            cell_traces.append(bout_signal)
        bout_dff.append(np.mean(cell_traces))

    antibouts_dff, antibouts_td_time = get_antibouts(
        _data.tracedata.traces)
    sponts, stdevs = get_sponts()

    # Calculate lick stats
    lickstats = pd.DataFrame(columns=[
        'File', 'Cell', 'Type'])
    lickstats_list = []
    cell_id = np.array(_data.tracedata.cells)

    for index, cell in enumerate(_data.tracedata.columns):
        if (bout_dff[index]) > ((sponts[index]) + ((stdevs[index]) * 2.58)):
            licktype = 'BOUT'
        elif (bout_dff[index]) < ((sponts[index]) - ((stdevs[index]) * 2.58)):
            licktype = 'ANTI-LICK'
        else:
            licktype = 'non-lick'

        lick_info = [
            session,
            cell,
            licktype
        ]
        lickstats_list.append(
            lick_info)

    for ind, d in enumerate(lickstats_list):
        track = len(lickstats)
        lickstats.loc[track] = d

    return lickstats


def get_stats(_data, session,
              output: Optional[bool] = False,
              results_dir: Optional[str] = '',
              return_data: bool = False
              ) -> Tuple[pd.DataFrame, 
                         pd.DataFrame,
                         pd.DataFrame]:
                         
    firstcell = _data.tracedata.traces.columns[1]
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

    raw_df = pd.DataFrame(columns=_data.tracedata.traces.columns)

    count = 0
    stats_list = []
    for cell in _data.tracedata.traces.columns[1:]:
        count += 1

        cell_list = []

        for stim, times in _data.eventdata.trial_times.items():
            ntrials = len(times)

            trial_list = []
            for iteration, trial in enumerate(times):

                # Index to analysis window
                data_ind = np.where(
                    (_data.tracedata.traces.time > trial) & (_data.tracedata.traces.time < trial + 5))[0]

                # Index to baseline window (current: 4s pre stim)
                bltime_ind = np.where(
                    (_data.tracedata.traces.time > trial - 4) & (_data.tracedata.traces.time < trial))[0]
                bltime = _data.tracedata.traces.time[bltime_ind]
                bltime = [min(bltime), max(bltime)]

                # Baseline statistics
                bl_signal = np.array(_data.tracedata.traces.iloc[bltime_ind, count])
                mean_bl = np.mean(bl_signal)
                stdev_bl = np.std(bl_signal)
                c_interv = mean_bl + stdev_bl * 2.58

                # Get calcium trace & time
                signal = np.array(_data.tracedata.traces.iloc[data_ind, count])

                # Get peak signal & time
                peak_signal = max(signal)
                peak_ts = _data.tracedata.traces.loc[_data.tracedata.traces[cell]
                                                     == peak_signal, 'Time(s)'].iloc[0]

                if peak_ts <= trial + 2.4:
                    peak_ts = max(bltime) + 2.4
                    shift = 'yes'
                else:
                    shift = 'no'

                # Get window 1s centered at peak
                peak_window_ind = func.get_peak_window(_data.tracedata.traces.time, peak_ts)
                time_lower = _data.tracedata.traces.time[peak_window_ind[0]]
                time_upper = _data.tracedata.traces.time[peak_window_ind[1]]
                response_window = np.array(_data.tracedata.traces.loc[
                                           peak_window_ind[0]:peak_window_ind[1], cell])

                window_ts = [time_lower, time_upper]
                mean_mag = np.mean(response_window)
                dff = ((mean_mag - mean_bl) / mean_bl)

                # if DF/F is significant, output concat new dataframe
                # with raw values from this trial
                if dff >= c_interv:

                    d = [stim, 'Trial {}'.format(iteration + 1)]
                    d.extend(
                        ['-' for _ in range(len(_data.tracedata.traces.columns[2:]))])

                    track = len(raw_df)

                    sig = 'Significant'
                    raw_df.loc[track] = d

                    baseline_df = (_data.tracedata.traces[_data.tracedata.traces['Time(s)'].between(
                        min(bltime), time_upper)])
                    raw_df = pd.concat([raw_df, baseline_df])

                else:
                    sig = 'ns'

                # Check if peak_signal is a duplicate
                for _ in signal:
                    func.dup_check(signal, peak_signal)

                # Pack stats into list
                iteration_list = [session, cell, stim,
                                  f'Trial {iteration + 1}',
                                  mean_bl, stdev_bl, window_ts,
                                  bltime, shift, dff, sig]
                trial_list.append(iteration_list)

            if cell == firstcell:
                summary_list = [
                    session,
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
        if not results_dir:
            raise AttributeError('Output called, but no directory filled in parameters.')

        import os

        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        print("Outputting data to Excel...")

        with pd.ExcelWriter(str(results_dir)
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

