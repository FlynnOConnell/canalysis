#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#process_data.py

Module(analysis): Stats class for PCA and general statistics/output.
"""
from __future__ import annotations

from typing import Optional, Iterable, Generator, Any, ClassVar

import numpy as np
import pandas as pd

from graphs.heatmaps import EatingHeatmap
from utils import funcs
from .analysis_utils import ca_pca
from .analysis_utils.analysis_funcs import map_colors


class ProcessData:
    def __init__(self, data,):
        self.data: ClassVar = data
        self.signals: pd.DataFrame = data.tracedata.signals
        self.zscores: pd.DataFrame = data.tracedata.zscores
        self.time: pd.Series | Iterable[Any] = data.tracedata.time
        self.cells: Iterable[Any] = data.tracedata.cells
        self.avgs: dict = data.nr_avgs
        self.session: str = data.filehandler.session
        self.trial_times: dict = data.eventdata.trial_times
        self.timestamps: dict = data.eventdata.timestamps
        self.antibouts = self.get_antibouts()
        self.sponts = self.get_sponts()
        self.ca_pca: ClassVar | None = None

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

    def get_taste_df(
            self
    ) -> Generator[Iterable, None, None]:
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

    # def loop_taste(
    #         self,
    #         save_dir: Optional[str] = "",
    #         cols: list = None,
    #         **kwargs
    # ) -> Generator[Iterable, None, None]:
    #     for stim, iteration, signal in self.get_taste_df():
    #         signal = signal[cols]
    #         hm = EatingHeatmap(
    #             title=f"{stim}," f" trial {iteration + 1}",
    #             cm="plasma",
    #             line_loc=10,
    #             save_dir=save_dir,
    #             _id=f"{stim}",
    #             **kwargs,
    #         ).single(signal.T)
    #         yield hm


