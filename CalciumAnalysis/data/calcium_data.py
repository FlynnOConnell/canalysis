#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#data.py

Module: Classes for data processing.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import ClassVar
from misc import funcs
from file_handling.file_handler import FileHandler

from all_data import AllData
from trace_data import TraceData
from taste_data import TasteData
from event_data import EventData
from graphs.graph_utils import Mixins
from misc import excepts as e
logger = logging.getLogger(__name__)


# %%

@dataclass
class CalciumData(Mixins.CalPlots):
    alldata: ClassVar[AllData] = AllData.Instance()

    filehandler: FileHandler
    tracedata: TraceData = field(init=False)
    eventdata: EventData = field(init=False)
    _tastedata: TasteData = field(init=False)

    def __post_init__(self):
        self.date = self.filehandler.date
        self.animal = self.filehandler.animal
        self.data_dir = self.filehandler.directory
        self.color_dict = self.filehandler.color_dict

        # Core
        self.tracedata: TraceData = self._set_tracedata()
        self.eventdata: EventData = self._set_eventdata()
        self._authenticate()

        self._tastedata: TasteData = TasteData(self.tracedata.signals,
                                               self.tracedata.time,
                                               self.eventdata.timestamps,
                                               self.color_dict)
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
        return EventData(self.filehandler)

    def _authenticate(self):
        if not isinstance(self.tracedata, TraceData):
            raise e.DataFrameError('Trace data must be a dataframe')
        if not isinstance(self.eventdata, EventData):
            raise e.DataFrameError('Event data must be a dataframe.')
        if not any(x in self.tracedata.signals.columns for x in ['C0', 'C00']):
            raise AttributeError("No cells found in DataFrame")
        return None

    def _add_instance(self):
        my_dict = type(self).alldata
        if self.keys_exist(my_dict, self.animal, self.date):
            logging.info(f'{self.animal}-{self.date} already exist.')
        elif self.keys_exist(my_dict, self.animal) \
                and not self.keys_exist(my_dict, self.date):
            my_dict[self.animal][self.date] = self
            logging.info(f'{self.animal} exists, {self.date} added.')
        elif not self.keys_exist(my_dict, self.animal):
            my_dict[self.animal] = {self.date: self}
            logging.info(f'{self.animal} and {self.date} added')
        return None
