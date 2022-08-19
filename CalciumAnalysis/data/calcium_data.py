#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#data.py

Module: Classes for data processing.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Generator, Iterable

import pandas as pd
import numpy as np

# matplotlib.use('Qt4Agg')
from scipy import stats
from .all_data import AllData
from .trace_data import TraceData
from .taste_data import TasteData
from .event_data import EventData
from .eating_data import EatingData
from .data_utils.file_handler import FileHandler
from graphs.graph_utils import Mixins
from utils import excepts as e
from utils import funcs

logger = logging.getLogger(__name__)


# %%


@dataclass
class CalciumData(Mixins.CalPlots):
    """
    General holder class for all trace/event related data, with additional
    storage for each session.
    """

    filehandler: FileHandler
    color_dict: color_dict
    adjust: Optional[int] = None
    tracedata: TraceData = field(init=False)
    eventdata: EventData = field(init=False)
    _tastedata: TasteData = field(init=False)

    alldata: ClassVar[AllData] = AllData.Instance()

    def __post_init__(self):
        self.date = self.filehandler.date
        self.animal = self.filehandler.animal
        self.data_dir = self.filehandler.directory
        self.session = self.filehandler.session

        # Core
        self.tracedata: TraceData = self._set_tracedata()
        self.eventdata: EventData = self._set_eventdata()
        if self.filehandler.eatingname is not None:
            self.eating_data = self._set_eatingdata()

        self.nr_avgs = self._get_nonreinforced_means()
        self._authenticate()

        self._tastedata: TasteData = TasteData(
            self.tracedata.signals,
            self.tracedata.time,
            self.eventdata.timestamps,
            self.color_dict,
        )
        self._add_instance()

    @classmethod
    def __len__(cls):
        return len(cls.alldata.keys())

    @staticmethod
    def keys_exist(element, *keys):
        return funcs.keys_exist(element, *keys)

    @property
    def tastedata(self):
        return self._tastedata

    @tastedata.setter
    def tastedata(self, **new_values):
        self._tastedata = TasteData(**new_values)

    def _set_tracedata(self) -> TraceData:
        return TraceData(self.filehandler)

    def _set_eventdata(self) -> EventData:
        return EventData(self.filehandler, self.color_dict)

    def _set_eatingdata(self) -> EatingData:
        return EatingData(self.filehandler, self.adjust)

    def _authenticate(self):
        if not isinstance(self.tracedata, TraceData):
            raise e.DataFrameError("Trace data must be a dataframe")
        if not isinstance(self.eventdata, EventData):
            raise e.DataFrameError("Event data must be a dataframe.")
        if not any(
            x in self.tracedata.signals.columns for x in ["C0", "C00", "C000", "C0000"]
        ):
            logging.debug(f"{self.tracedata.signals.head()}")
            raise AttributeError(
                f"No cells found in DataFrame: " f"{self.tracedata.signals.head()}"
            )
        return None

    def _add_instance(self):
        my_dict = type(self).alldata
        if self.keys_exist(my_dict, self.animal, self.date):
            logging.info(f"{self.animal}-{self.date} already exist.")
        elif self.keys_exist(my_dict, self.animal) and not self.keys_exist(
            my_dict, self.date
        ):
            my_dict[self.animal][self.date] = self
            logging.info(f"{self.animal} exists, {self.date} added.")
        elif not self.keys_exist(my_dict, self.animal):
            my_dict[self.animal] = {self.date: self}
            logging.info(f"{self.animal} and {self.date} added")
        return None

    @staticmethod
    def reorder(df: pd.DataFrame, cols: list):
        return df[cols]

    def _get_nonreinforced_means(self) -> dict:
        """
        Get a dictionary (Cell: mean) of means for non-reinforced lick events.
        Returns
        -------
        Cell: mean dictionary.
        """
        ev_time = funcs.get_matched_time(
            self.tracedata.time, self.eventdata.nonreinforced
        )
        nr_signal = self.tracedata.zscores.loc[
            self.tracedata.zscores["time"].isin(ev_time)
        ].drop("time", axis=1)
        avgs = {}
        for column in nr_signal.columns:
            mean = (nr_signal[column]).mean()
            avgs[column] = mean
        return avgs

    def get_signal_bycell(self, i):
        """return a list of signal values via cell integer indexing (0 through N cells"""
        return list(self.tracedata.signals.iloc[:, i])

    def get_signal_bytime(self, start, stop, zscores: bool = True):
        """
        Fetch signals between two timepoints.
        Parameters
        ----------
        zscores :
        start : beginning of slice
        stop : end of slice
        zscores : whether to return zscore signals

        Returns
        -------
        pd.DataFrame of sliced signals given start:stop
        """
        if isinstance(start, list):
            start = start[-1]
        if isinstance(stop, list):
            stop = stop[-1]
        if zscores:
            return self.tracedata.zscores.iloc[start:stop]
        else:
            return self.tracedata.tracedata.iloc[start:stop]

    def get_eating_signals(self,) -> Generator[Iterable, None, None]:
        data = self.eating_data.eatingdata.to_numpy()
        size = len(data)
        counter = 0
        for index, x in (enumerate(data)):
            if index > (len(data)-2):
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
                    yield signal, counter, entry_start, entry_end, eating_end

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
                    yield signal, counter, entry_start, entry_end, eating_end

