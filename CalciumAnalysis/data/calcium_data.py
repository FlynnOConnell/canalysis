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

        # Core
        self.tracedata: TraceData = self._set_tracedata()
        self.eventdata: EventData = self._set_eventdata()
        if self.filehandler.eatingname is not None:
            self.eating_data = self._set_eatingdata()
            self.eating_signals = self.__split_data()
            self.eating_time = self.eating_signals.pop('time')
            self.eating_zscores = self.__set_eating_zscores()

        self.nr_avgs = self._get_nonreinforced_signals()
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

    def _get_nonreinforced_signals(self) -> dict:
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
        return list(self.tracedata.signals.iloc[:, i])

    def get_signal_bytime(self, start, stop, zscores: bool = True):
        if isinstance(start, list):
            start = start[-1]
        if isinstance(stop, list):
            stop = stop[-1]
        if zscores:
            return self.tracedata.zscores.iloc[start:stop]
        else:
            return self.tracedata.tracedata.iloc[start:stop]

    def __split_data(self, ):
        last = funcs.get_matched_time(
            self.tracedata.time, self.eventdata.timestamps['Lick'][-1], return_index=True, single=True
        )
        return self.tracedata.tracedata.iloc[last:, :]

    def __set_eating_zscores(self) -> pd.DataFrame:
        eating_zscores_df = pd.DataFrame(columns=self.eating_signals.columns)
        eating_signals = self.eating_signals.copy()
        for cell in eating_signals.columns:
            zscore = np.array(stats.zscore(self.eating_signals[cell]))
            new_z = np.where(zscore < 0, 0, zscore)
            eating_zscores_df[cell] = new_z
        return eating_zscores_df

    def get_eating_signals(self) -> Generator[Iterable, None, None]:
        data = self.eating_zscores.to_numpy()
        for x in data:
            time_start = funcs.get_matched_time(self.eating_time, x[1], return_index=True, single=True)
            time_end = funcs.get_matched_time(self.eating_time, x[2], return_index=True, single=True)
            signal = self.eating_zscores.iloc[time_start:time_end]
            yield signal, x[0]

