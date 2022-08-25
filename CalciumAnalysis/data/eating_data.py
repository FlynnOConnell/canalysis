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
from typing import Optional, Generator, Iterable, ClassVar
from .data_utils.file_handler import FileHandler
from data.trace_data import TraceData
logger = logging.getLogger(__name__)


@dataclass
class EatingData:
    filehandler: FileHandler
    tracedata: TraceData
    adjust: Optional[int] = 34
    eatingdata: pd.DataFrame = field(init=False)
    eatingsignals: pd.DataFrame = field(init=False)

    def __post_init__(self,):
        self.eatingdata = self.filehandler.get_eatingdata().sort_values("TimeStamp")
        # Core attributes
        self.set_adjust()
        self.__clean()
        self.eatingsignals: pd.DataFrame = self._set_eating_signals()

    def __hash__(self,):
        return hash(repr(self))

    def set_adjust(self,) -> None:
        for column in self.eatingdata.columns[1:]:
            self.eatingdata[column] = self.eatingdata[column] + self.adjust

    def __clean(self,) -> None:
        self.eatingdata.drop(
            ["Marker Type", "Marker Event Id", "Marker Event Id 2", "value1", "value2"],
            axis=1,
            inplace=True,
        )
        self.eatingsignals = self.eatingdata.loc[
            self.eatingdata["Marker Name"].isin(["Entry", "Eating", "Grooming"])
        ]

    def _set_eating_signals(self):
        aggregate_eating_signals = pd.DataFrame()
        for signal, event in self.generate_signals():
            aggregate_eating_signals = pd.concat(
                [aggregate_eating_signals, signal],
                axis=1
            )
        return aggregate_eating_signals

    def generate_signals(
            self,
    ) -> Generator[(pd.DataFrame, str), None, None]:
        for x in self.eatingdata.to_numpy():
            signal = (self.tracedata.zscores.iloc[
                      funcs.get_matched_time(
                          self.tracedata.time, x[1], return_index=True, single=True
                      ):funcs.get_matched_time(
                          self.tracedata.time, x[2], return_index=True, single=True
                      )]).drop(columns=['time'])
            yield signal, x[0]

    def generate_entry_eating_signals(
            self,
    ) -> Generator[Iterable, None, None]:
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
                    entry_start = funcs.get_matched_time(
                        self.tracedata.time, x[1], return_index=True, single=True
                    )
                    entry_end = funcs.get_matched_time(
                        self.tracedata.time, x[2], return_index=True, single=True
                    )
                    eating_end = funcs.get_matched_time(
                        self.tracedata.time, data[nxt][2], return_index=True, single=True
                    )
                    signal = (self.tracedata.zscores.iloc[
                              entry_start:eating_end]).drop(columns=['time'])
                    yield signal, x[0], counter, entry_start, entry_end, eating_end

                elif data[nxt][0] == 'Eating' and data[nxt2][0] == 'Eating':
                    entry_start = funcs.get_matched_time(
                        self.tracedata.time, x[1], return_index=True, single=True
                    )
                    entry_end = funcs.get_matched_time(
                        self.tracedata.time, x[2], return_index=True, single=True
                    )
                    eating_end = funcs.get_matched_time(
                        self.tracedata.time, data[nxt2][2], return_index=True, single=True
                    )
                    signal = (self.tracedata.zscores.iloc[
                              entry_start:eating_end]).drop(columns=['time'])
                    yield signal, x[0], counter, entry_start, entry_end, eating_end

