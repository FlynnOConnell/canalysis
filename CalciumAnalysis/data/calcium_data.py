#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#data.py

Module: Classes for data processing.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import pandas as pd

from containers.all_data import AllData
from containers.trace_data import TraceData
from containers.taste_data import TasteData
from containers.event_data import EventData
from containers.eating_data import EatingData
from data_utils.file_handler import FileHandler
from graphs.graph_utils import Mixins
from utils import excepts as e
from utils import funcs

logger = logging.getLogger(__name__)


@dataclass
class CalciumData(Mixins.CalPlots):
    """
    General holder class for all trace/event related data, with additional
    storage for each session.
    """

    __filehandler: FileHandler
    color_dict: dict
    adjust: Optional[int] | None = None
    tracedata: TraceData = field(init=False)
    eventdata: EventData = field(init=False)
    tastedata: TasteData = field(init=False)
    alldata: ClassVar[AllData] = AllData.Instance()

    def __post_init__(self):

        # Instance info
        self.date = self.__filehandler.date
        self.animal = self.__filehandler.animal
        self.data_dir = self.__filehandler.directory
        self.session = self.__filehandler.session

        # Core data
        self.tracedata: TraceData = TraceData(self.__filehandler)
        self.eventdata: EventData = EventData(self.__filehandler, self.color_dict, self.tracedata.time)
        if self.__filehandler.eatingname is not None:
            self.eatingdata: EatingData = EatingData(
                self.__filehandler,
                self.tracedata,
                self.color_dict
            )
        self.nr_avgs = self._get_nonreinforced_means()
        self._authenticate()

        self.tastedata: TasteData = TasteData(
            self.tracedata.zscores,
            self.tracedata.time,
            self.eventdata.timestamps,
            self.eventdata.trial_times,
            self.nr_avgs,
            self.color_dict
        )
        self._add_instance()

    @classmethod
    def __len__(cls):
        return len(cls.alldata.keys())

    def __repr__(self):
        return type(self).__name__

    @staticmethod
    def keys_exist(element, *keys):
        return funcs.keys_exist(element, *keys)

    @property
    def size(self):
        return len(self.tracedata.cells)

    def reset_tastedata(self):
        self.tastedata = TasteData(
                    self.tracedata.zscores,
                    self.tracedata.time,
                    self.eventdata.timestamps,
                    self.eventdata.trial_times,
                    self.nr_avgs,
                    self.color_dict
                )

    def _authenticate(self):
        if not isinstance(self.tracedata, TraceData):
            raise e.DataFrameError("Trace data must be a dataframe")
        if not isinstance(self.eventdata, EventData):
            raise e.DataFrameError("Event data must be a dataframe.")
        if not any(
                x in self.tracedata.signals.columns for x in
                ["C0", "C00", "C000", "C0000"]
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

    def _get_nonreinforced_means(
            self,
    ) -> dict:
        """
        Get a dictionary (Cell: mean) of means for non-reinforced lick events.
        Returns
        -------
        Cell: mean dictionary.
        """
        nr_signal = self.tracedata.zscores.loc[
            self.tracedata.zscores["time"].isin(self.eventdata.nonreinforced)
        ].drop("time", axis=1)
        avgs = {}
        for column in nr_signal.columns:
            mean = (nr_signal[column]).mean()
            avgs[column] = mean
        return avgs

    def get_signal_bycell(
            self,
            i
    ) -> list[pd.Series]:
        """return a list of signal values via cell integer indexing (0 through N cells"""
        return list(self.tracedata.signals.iloc[:, i])

    def combine(self, eatingevent: list = None, stimevent: list = None) -> tuple[pd.DataFrame, pd.Series]:
        eating, eatingcolors = self.eatingdata.get_signals_from_events(eatingevent)
        stim, stimcolors = self.tastedata.get_signals_from_events(stimevent)
        return pd.concat([eating, stim], axis=0), pd.concat([eatingcolors, stimcolors], axis=0)
