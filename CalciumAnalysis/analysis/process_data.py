#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#process_data.py

Module(analysis): Stats class for PCA and general statistics/output.
"""
from __future__ import annotations

import logging
from typing import Optional, Iterable, Generator

import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import stats

from data.calcium_data import CalciumData
from graphs.heatmaps import Heatmap
from utils import funcs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ProcessData:
    def __init__(self, data, outpath=None):
        assert isinstance(data, CalciumData)
        self.outpath = outpath
        if isinstance(outpath, str):
            self.outpath = Path(outpath)
        self.data = data
        self.signals = data.tracedata.signals
        self.zscores = data.tracedata.zscores
        self.time = data.tracedata.time
        self.cells = data.tracedata.cells
        self.avgs = data.nr_avgs
        self.session = data.filehandler.session
        self.trial_times = data.eventdata.trial_times
        self.timestamps = data.eventdata.timestamps
        self.antibouts = self.get_antibouts()
        self.sponts = self.get_sponts()

    @staticmethod
    def principal_components(df: pd.DataFrame, numcomp: Optional[int] = 2):

        data = StandardScaler().fit_transform(df)
        pca = PCA(n_components=numcomp)
        data_fit = pca.fit_transform(data)

        variance_ns = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        labels = [
            "PC" + str(x) + f" - {variance_ns[x - 1]}%"
            for x in range(1, len(variance_ns) + 1)
        ]
        df_ns = pd.DataFrame(data_fit, columns=labels)

        return df_ns, variance_ns

    def get_antibouts(self) -> pd.DataFrame | pd.Series:
        antibouts = pd.DataFrame()
        for interv in funcs.interval(self.timestamps["Lick"], gap=10, outer=True):
            df = self.signals.loc[(self.time > (interv[0])) & (self.time < (interv[1]))]
            antibouts = pd.concat([antibouts, df], axis=0)
        return antibouts

    def get_sponts(self) -> pd.DataFrame | pd.Series:
        sponts = pd.DataFrame()
        for interv in funcs.interval(self.timestamps["Lick"], gap=30, outer=True):
            df = self.signals[(self.time > (interv[0])) & (self.time < (interv[1]))]
            sponts = pd.concat([sponts, df], axis=0)
        return sponts

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
                    # Check if peak_signal is a duplicate
                    for _ in signal:
                        funcs.dup_check(signal, peak_signal)
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

    def get_taste_df(self) -> Generator[Iterable, None, None]:

        signals = self.zscores.copy().drop('time', axis=1)
        for cell in signals:
            signals[cell] = (signals[cell] - self.avgs[cell])
            # Replace negatives with 0 using numpys fancy indexing
            signals[cell][signals[cell] < 0] = 0

        for stim, times in self.trial_times.items():
            for iteration, trial in enumerate(times):
                data_ind = np.where((self.time > trial - 1) & (self.time < trial + 3))[0]
                signal = signals.iloc[data_ind, 1:]
                yield stim, iteration, signal

    def loop_taste(self,
                   save_dir: Optional[str] = '',
                   **kwargs
                   ) -> Generator[Iterable, None, None]:
        for stim, iteration, signal in self.get_taste_df():
            if stim in ['Chocolate', 'Sucrose']:
                hm = Heatmap(
                    title=f"{stim},"
                          f" trial {iteration}",
                    cm='plasma',
                    line_loc=10,
                    save_dir=save_dir,
                    _id=f'{stim}',
                    **kwargs
                ).single(signal.T)
                yield hm

    def loop_eating(
            self,
            save_dir: Optional[str] = '',
            **kwargs
    ) -> Generator[Iterable, None, None]:
        for signal, event, starttime, endtime in self.data.get_eating_signals():
            start = starttime
            end = endtime
            xlabel = end - start
            hm = Heatmap(
                title=f'{event}',
                xlabel=xlabel,
                cm='plasma',
                save_dir=save_dir,
                _id=f'{event}_{start}',
                **kwargs
            ).single(signal.T)
            yield hm

    def get_event_df(
            self,
    ):
        df_eating = pd.DataFrame(columns=self.data.tracedata.signals.columns)
        df_entry = pd.DataFrame(columns=self.data.tracedata.signals.columns)
        df_grooming = pd.DataFrame(columns=self.data.tracedata.signals.columns)
        for signal, event, _, _ in self.data.get_eating_signals():
            if event == 'Grooming':
                pd.concat([df_eating, signal], axis=1)
            elif event == 'Entry':
                pd.concat([df_entry, signal], axis=1)
            else:
                pd.concat([df_entry, signal], axis=1)
        return df_eating, df_entry, df_grooming





