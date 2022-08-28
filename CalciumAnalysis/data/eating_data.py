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
from typing import Optional, Generator, Iterable, ClassVar
from data_utils.file_handler import FileHandler
from data.trace_data import TraceData

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

    def get_time_index(self, num: int | float):
        """Return INDEX where tracedata time matches argument num."""
        return self.tracedata.time[self.tracedata.time == num].index[0]

    def __set_adjust(self, ) -> None:
        for column in self.eatingdata.columns[1:]:
            self.eatingdata[column] = self.eatingdata[column] + self.adjust

    def __clean(self, ) -> None:
        self.eatingdata.drop(
            ["Marker Type", "Marker Event Id", "Marker Event Id 2", "value1", "value2"],
            axis=1,
            inplace=True,
        )
        self.eatingdata = self.eatingdata.loc[
            self.eatingdata["Marker Name"].isin(["Entry", "Eating", "Grooming"])
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
        for x in self.eatingdata.to_numpy():
            signal = (self.tracedata.zscores.iloc[
                     np.where(self.tracedata.time == (x[1]))[0][0]:
                     np.where(self.tracedata.time == (x[2]))[0][0]
                      ]).drop(columns=['time'])
            yield signal, x[0]

    def generate_entry_eating_signals(
            self,
    ) -> Generator[Iterable, None, None]:
        """ Generator for eating events, with entry and eating in one interval."""
        data = self.eatingdata.to_numpy()
        counter = 0
        for index, x in (enumerate(data)):
            if index > (len(data) - 2):
                break
            if x[0] == 'Entry':
                counter += 1
                nxt = index + 1
                nxt2 = index + 2
                if data[nxt][0] == 'Eating' and data[nxt2][0] == 'Entry':
                    signal = (self.tracedata.zscores.iloc[
                              np.where(self.tracedata.time == (x[1]))[0][0]:
                              np.where(self.tracedata.time == (x[2]))[0][0]
                              ]).drop(columns=['time'])
                    # signal, event, counter, entry start, entry end, eating end
                    yield signal, x[0], counter, x[1], x[2], data[nxt][2]

                elif data[nxt][0] == 'Eating' and data[nxt2][0] == 'Eating':
                    signal = (self.tracedata.zscores.iloc[
                              np.where(self.tracedata.time == (x[1]))[0][0]:
                              np.where(self.tracedata.time == data[nxt][2])[0][0]
                              ]).drop(columns=['time'])
                    yield signal, x[0], counter, x[1], x[2], data[nxt][2]

    def get_eating_df(
            self,
    ) -> pd.DataFrame:
        """
        Return a dataframe of eating, grooming and entry events.
        Containes 'events' column.
        """
        df_eating = pd.DataFrame()
        df_entry = pd.DataFrame()
        df_grooming = pd.DataFrame()
        for signal, event in self.generate_signals():
            if event == "Grooming":
                df_grooming = pd.concat([df_grooming, signal], axis=0)
                df_grooming['events'] = 'grooming'
            if event == "Entry":
                df_entry = pd.concat([df_entry, signal], axis=0)
                df_entry['events'] = 'entry'
            if event == 'Eating':
                df_eating = pd.concat([df_eating, signal], axis=0)
                df_eating['events'] = 'eating'
        return pd.concat([df_eating, df_grooming, df_entry], axis=0)
