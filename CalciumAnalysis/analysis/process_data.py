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
from .analysis_utils import analysis_funcs
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


    def get_taste_df(self) -> Generator[Iterable, None, None]:
        signals = self.zscores.copy().drop("time", axis=1)
        for cell in signals:
            signals[cell] = signals[cell] - self.avgs[cell]
            # Replace negatives with 0 using numpys fancy indexing
            signals[cell][signals[cell] < 0] = 0

        for stim, times in self.trial_times.items():
            for iteration, trial in enumerate(times):
                data_ind = np.where((self.time > trial) & (self.time < trial + 6))[
                    0
                ]
                signal = signals.iloc[data_ind, :]
                yield stim, iteration, signal


    def loop_taste(
            self,
            save_dir: Optional[str] = "",
            cols: list = None,
            **kwargs
    ) -> Generator[Iterable, None, None]:
        for stim, iteration, signal in self.get_taste_df():
            signal = signal[cols]
            hm = Heatmap(
                title=f"{stim}," f" trial {iteration + 1}",
                cm="plasma",
                line_loc=10,
                save_dir=save_dir,
                _id=f"{stim}",
                **kwargs,
            ).single(signal.T)
            yield hm

    def loop_eating(
        self,
        save_dir: Optional[str] = "",
        cols: list = None,
        **kwargs
    ) -> Generator[Iterable, None, None]:
        for signal, counter, starttime, middle, endtime in self.data.get_eating_signals():
            for cell in signal:
                signal[cell][signal[cell] < 0] = 0
            xlabel = (endtime - starttime) * 0.1
            if cols:
                signal = signal[cols]
            hm = Heatmap(
                title=f"Trial: {counter}",
                xlabel=f'{np.round(xlabel, 2)} seconds',
                cm="plasma",
                save_dir=save_dir,
                _id=f"{counter}",
                line_loc=middle-starttime,
                **kwargs,
            ).single(signal.T)
            yield hm

    def get_event_df(self,):
        df_eating = pd.DataFrame()
        df_entry = pd.DataFrame()
        df_grooming = pd.DataFrame()
        for signal, event, _, _ in self.data.get_eating_signals():
            if event == "Grooming":
                df_grooming = pd.concat([df_grooming, signal], axis=0)
                df_grooming['event'] = 'grooming'
            if event == "Entry":
                df_entry = pd.concat([df_entry, signal], axis=0)
                df_entry['event'] = 'entry'
            if event == 'Eating':
                pd.concat([df_eating, signal], axis=0)
                df_eating['event'] = 'eating'
        return pd.concat([df_eating, df_grooming, df_entry], axis=0)

    def get_pca(self,):
        df_pca = analysis_funcs.principal_components(self.data.tracedata.signals)
        