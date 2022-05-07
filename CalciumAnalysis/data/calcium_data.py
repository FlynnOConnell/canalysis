#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#data.py

Module: Classes for data processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from misc import funcs
from misc.file_handler import FileHandler

from data_utils.all_data import AllData
from data_utils.trace_data import TraceData
from data_utils.taste_data import TasteData
from data_utils.event_data import EventData
from graphs.graph_utils import Mixins
from misc import excepts as e

logger = logging.getLogger(__name__)


# %%

# Storing instances in a mutable mapping from abstract base class
# for some extra functionality in how we iterate, count and represent
# items in the dict.


@dataclass
class CalciumData(Mixins.CalPlots):
    alldata = AllData.Instance()
    color_dict = {
        'ArtSal': 'dodgerblue',
        'MSG': 'darkorange',
        'NaCl': 'lime',
        'Sucrose': 'magenta',
        'Citric': 'yellow',
        'Quinine': 'red',
        'Rinse': 'lightsteelblue',
        'Lick': 'darkgray'
    }

    def __init__(
            self,
            animal: str,
            date: str,
            data_dir: str,
            **kwargs: Optional[dict]
    ):

        # Update the internal dict with kw arguments
        self.filehandler = self.get_handler(animal, date, data_dir)
        self.date = None
        self.animal = None
        self.data_dir = None
        self.__dict__.update(kwargs)

        # Core
        self.tracedata: TraceData = self._set_tracedata()
        self.eventdata: EventData = self._set_eventdata()
        self._authenticate()

        self.color_dict = CalciumData.color_dict
        self._tastedata: TasteData = TasteData(self.tracedata.signals,
                                               self.tracedata.time,
                                               self.eventdata.timestamps,
                                               self.color_dict)
        self._add_instance()
        # TODO: Replace self.color_dict ref with CalciumData.color_dict

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self.date == other.date and self.animal == other.animal

    @classmethod
    def __len__(cls):
        return len(cls.alldata.keys())

    @staticmethod
    def get_handler(animal, date, data_dir):
        return FileHandler(animal, date, data_dir)

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
