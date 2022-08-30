#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#eating_data.py

Module: Classes for food-related data processing.
"""

from __future__ import annotations
from utils import funcs
import logging
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from typing import Optional, Generator, Iterable
from data_utils.file_handler import FileHandler
from data.trace_data import TraceData
from utils.wrappers import log_time

logger = logging.getLogger(__name__)


@dataclass
class EatingData:
    filehandler: FileHandler
    tracedata: TraceData
    adjust: Optional[int] = 34
    eatingdata: pd.DataFrame = field(init=False)
    eatingsignals: pd.DataFrame = field(init=False)

    def __post_init__(self, ):
        self.eatingdata = self.filehandler.get_eatingdata().sort_values("TimeStamp")
        # Core attributes
        self.__set_adjust()
        self.__clean()
        self.__match()
        self.eatingsignals: pd.DataFrame = self.__set_eating_signals()

    def __repr__(self):
        return type(self).__name__

    def __hash__(self, ):
        return hash(repr(self))

    def get_time_index(self, time: int | float, ):
        """Return INDEX where tracedata time matches argument num."""
        return np.where(self.tracedata.time == time)[0][0]

    def get_signal_zscore(self, start: int | float, stop: int | float, ):
        """Return  where tracedata time matches argument num."""
        return self.tracedata.zscores.iloc[
               self.get_time_index(start):
               self.get_time_index(stop)
               ].drop(columns=['time'])

    def __set_adjust(self, ) -> None:
        for column in self.eatingdata.columns[1:]:
            self.eatingdata[column] = self.eatingdata[column] + self.adjust

    def __clean(self, ) -> None:
        self.eatingdata = self.eatingdata.loc[
            self.eatingdata["Marker Name"].isin(["Entry", "Eating", "Grooming", "Approach", "Interval"])
        ]

    def __match(self, ):
        self.eatingdata['TimeStamp'] = funcs.get_matched_time(
            self.tracedata.time, self.eatingdata['TimeStamp'])
        self.eatingdata['TimeStamp2'] = funcs.get_matched_time(
            self.tracedata.time, self.eatingdata['TimeStamp2'])

    def __set_eating_signals(self):
        aggregate_eating_signals = pd.DataFrame()
        for signal, event in self.generate_signals():
            signal['event'] = event
            aggregate_eating_signals = pd.concat(
                [aggregate_eating_signals, signal],
                axis=0
            )
        return aggregate_eating_signals.sort_index()

    def generate_signals(
            self,
    ) -> Generator[(pd.DataFrame, str), None, None]:
        """Generator for each eating signal."""
        return ((self.get_signal_zscore(x[1], x[2]), x[0]) for x in self.eatingdata.to_numpy())

    def generate_entry_eating_signals(
            self,
    ) -> Generator[Iterable, None, None]:
        """ Generator for eating events, with entry and eating in one interval."""
        data = self.eatingdata.to_numpy()
        counter = 0
        for index, x in (enumerate(data)):
            counter += 1
            nxt = index + 1
            nxt2 = index + 2
            if index > (len(data) - 2):
                break
            if x[0] == 'Approach' and data[nxt][0] == 'Entry' and data[nxt2][0] == 'Eating':
                # yield: signal, counter, approach start, entry start, eating start, eating end
                yield self.get_signal_zscore(x[1], data[nxt2][2]), counter, x[1], data[nxt][1], data[nxt2][1], \
                      data[nxt2][2]

    def get_eating_df(
            self,
            events: list
    ) -> pd.DataFrame:
        """
        Return a dataframe of eating, grooming and entry events.
        Containes 'events' column.
        """
        df_interval = pd.DataFrame()
        df_grooming = pd.DataFrame()
        for signal, event in self.generate_signals():
            if event == [events]:
                df_grooming = pd.concat([df_grooming, signal], axis=0)
                df_grooming['events'] = 'grooming'
            if event == "Interval":
                df_interval = pd.concat([df_interval, signal], axis=0)
                df_interval['events'] = 'interval'
        return pd.concat([df_interval, df_grooming], axis=0)

    def baseline(self):
        data = self.eatingdata.to_numpy()
        baseline = data[np.where(data[:, 0] == 'Interval')[0][0]]
        signal = self.get_signal_zscore(baseline[1], baseline[2])
        return signal

    def loop_eating(
            self,
            save_dir: Optional[str] = "",
            cols: list = None,
            **kwargs
    ) -> Generator[Iterable, None, None]:
        from graphs.heatmaps import EatingHeatmap
        for signal, counter, approachstart, entrystart, eatingstart, eatingend in self.generate_entry_eating_signals():
            for cell in signal:
                signal[cell][signal[cell] < 0] = 0
            if cols:
                signal = signal[cols]
            heatmap = EatingHeatmap(
                signal.T,
                title="Approach, Entry and Eating Interval",
                save_dir=save_dir,
            )
            heatmap.interval_heatmap(eatingstart, entrystart, eatingend)
            heatmap.show_heatmap()
            yield heatmap

