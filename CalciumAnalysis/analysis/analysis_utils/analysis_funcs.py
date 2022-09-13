#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#analysis_funcs.py

Module(analysis_utils): Functions to assist in data analysis
"""
from __future__ import annotations

import logging
from typing import Optional, Any, Iterable
import numpy as np
import pandas as pd
from scipy.stats import stats
from utils import funcs


def map_colors(
    iterable: Iterable[Any],
    colors_dict: dict = None
) -> list:
    """
    Given dataframe and dictionary, maps value of dictionary to event of dataframe column.
    Parameters
    """
    if not colors_dict:
        colors_dict = {
            'grooming': 'green',
            'entry': 'blue',
            'eating': 'red'
        }
    return [colors_dict[x] for x in iterable]



def get_stats(self) -> pd.DataFrame | None:
    stats_df = pd.DataFrame(
            columns=[
                "File",
                "Cell",
                "Stimulus",
                "Trial",
                "Baseline (mean)",
                "Baseline (st_dev)",
                "Signal Timestamps (start, stop)",
                "Baseline Timestamps (start, stop)",
                "Shifted?",
                "deltaF/F",
                "Significant?",
            ]
    )
    summary = pd.DataFrame(columns=["File", "Stimulus", "Num Trials"])
    raw_df = pd.DataFrame(columns=self.cells)
    count = 0
    stats_list = []
    for cell in self.signals.columns:
        count += 1
        cell_list = []
        for stim, times in self.trial_times.items():
            ntrials = len(times)
            trial_list = []
            for iteration, trial in enumerate(times):
                # Index to analysis window
                data_ind = np.where((self.time > trial) & (self.time < trial + 5))[
                    0
                ]
                # Index to baseline window (current: 4s pre stim)
                bltime_ind = np.where(
                        (self.time > trial - 4) & (self.time < trial)
                )[0]
                bltime = self.time[bltime_ind]
                bltime = [min(bltime), max(bltime)]
                # Baseline statistics
                bl_signal = np.array(self.signals[bltime_ind, count])
                mean_bl = np.mean(bl_signal)
                stdev_bl = np.std(bl_signal)
                c_interv = mean_bl + stdev_bl * 2.58
                # Get calcium trace & time
                signal = np.array(self.signals[data_ind, count])
                # Get peak signal & time
                peak_signal = max(signal)
                peak_ts = self.signals[
                    self.signals[cell] == peak_signal, "time"
                ].iloc[0]
                if peak_ts <= trial + 2.4:
                    peak_ts = max(bltime) + 2.4
                    shift = "yes"
                else:
                    shift = "no"
                # Get window 1s centered at peak
                peak_window_ind = funcs.get_peak_window(self.time, peak_ts)
                time_lower = self.time[peak_window_ind[0]]
                time_upper = self.time[peak_window_ind[1]]
                response_window = np.array(
                        self.signals[peak_window_ind[0]: peak_window_ind[1], cell]
                )
                window_ts = [time_lower, time_upper]
                mean_mag = np.mean(response_window)
                dff = (mean_mag - mean_bl) / mean_bl
                # if DF/F is significant, output concat new dataframe
                # with raw values from this trial
                if dff >= c_interv:
                    d = [stim, "Trial {}".format(iteration + 1)]
                    d.extend(["-" for _ in range(len(self.signals[1:]))])
                    track = len(raw_df)
                    sig = "Significant"
                    raw_df.loc[track] = d
                    baseline_df = self.signals[
                        self.time.between(min(bltime), time_upper)
                    ]
                    raw_df = pd.concat([raw_df, baseline_df])
                else:
                    sig = "ns"
                # Pack analysis into list
                iteration_list = [
                    self.session,
                    cell,
                    stim,
                    f"Trial {iteration + 1}",
                    mean_bl,
                    stdev_bl,
                    window_ts,
                    bltime,
                    shift,
                    dff,
                    sig,
                ]
                trial_list.append(iteration_list)
            if cell == self.signals[0]:
                summary_list = [self.session, stim, ntrials]
                track = len(summary)
                summary.loc[track] = summary_list
            cell_list.append(trial_list)
        stats_list.append(cell_list)
    flaten_stats = funcs.flatten(funcs.flatten(stats_list))
    # Make dataframe from flattened list of analysis
    for ind, d in enumerate(flaten_stats):
        track = len(stats_df)
        stats_df.loc[track] = d
    logging.info("Stats successfully completed.")

    if self.outpath:
        print("Outputting data to Excel...")
        with pd.ExcelWriter(self.outpath / "/" / "_statistics.xlsx") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
            # lickstats.to_excel(
            #     writer, sheet_name='Lick cells', index=False)
            stats_df.to_excel(writer, sheet_name="Trial Stats", index=False)
        print(" statistics successfully transferred to Excel!")
        return None
    else:
        return stats_df
