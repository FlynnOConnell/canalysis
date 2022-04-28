#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#data.py

Module: Classes for data processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Type, Optional, Iterable, Callable
import scipy.stats as stats
from collections.abc import MutableMapping

import numpy as np
import pandas as pd

from core.core_utils.file_handler import FileHandler
from core.core_utils import funcs as func
from utils.wrappers import Singleton
from utils import excepts as e
from graphs.graph_utils import Mixins

logger = logging.getLogger(__name__)

# %%

# Storing instances in a mutable mapping from abstract base class
# for some extra functionality in how we iterate, count and represent
# items in the dict.
@Singleton
class AllData(MutableMapping):
    """
    Custom mapping that works with properties from mutable object:
        
    alldata = AllData(function='xyz')
    and 
    d.function returns 'xyz'
    """

    def __init__(self, *args, **kwargs):
        # Update the class dict atributes
        self.__dict__.update(*args, **kwargs)

    # Standard dict methods, no overloading needed
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    # These are what we want to change
    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return "\n".join(f"{key} - {len(value)} sessions." for key, value in self.__dict__.items())

    def __repr__(self):
        return "\n".join(f"{key} - {len(value)} sessions." for key, value in self.__dict__.items())


def set_params():
    from matplotlib import rcParams
    rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": 'bold',
        'axes.facecolor': 'w',
        "axes.labelsize": 10,
        "lines.linewidth": 1,
        "savefig.dpi": 300,
    })


@dataclass
class CalciumData(Mixins.CalPlots):
    alldata = AllData.instance()
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

    def __init__(self,
                 animal: str,
                 date: str,
                 data_dir: str,
                 clean: Optional[bool] = True,
                 **kwargs: Optional[dict]
                 ):

        # Update the internal dict with kw arguments
        self.__dict__.update(kwargs)

        # Session information
        self.animal = animal
        self.date = date
        self.session = animal + '_' + date
        self.data_dir = data_dir
        self.filehandler = self.get_handler()

        # Core
        self.tracedata: Type[pd.DataFrame]
        self.eventdata: Type[pd.DataFrame]

        self._get_data()
        self._authenticate()

        # Core attributes
        self.signals = self._set_trace_signals()
        self.cells = np.array(self.tracedata.columns[1:])
        self.time = self.tracedata['Time(s)']
        self.binsize = self.time[2] - self.time[1]
        self.zscores = self._get_zscores()

        # Event attributes
        self.timestamps = {}
        self.trial_times = {}
        self.allstim = []
        self.drylicks = []
        self.licktime: Iterable
        self.numlicks: int | None = None
        self._set_event_attrs()
        self.color_dict = CalciumData.color_dict
        self._add_instance()

        # logging.info(f'Data instance for: \n {self.animal} - {self.date} \n ...instantiated.')

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self.date == other.date and self.animal == other.animal

    @classmethod
    def __len__(cls):
        return len(cls.alldata.keys())

    def get_handler(self):
        return FileHandler(self.data_dir, self.animal, self.date)

    @staticmethod
    def keys_exist(element, *keys):
        return func.keys_exist(element, *keys)

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

    def get_trials(self) -> None:
        for stim, trials in self.trial_times.items():
            logging.info(f'{stim} - {len(trials)}')
        return None

    def _get_zscores(self) -> pd.DataFrame:

        zscores = pd.DataFrame(columns=self.signals.columns)
        for cell in self.signals.columns:
            allsignal = self.signals[cell]
            zscore = pd.Series(stats.zscore(allsignal))
            zscores[cell] = zscore
        zscores['Time(s)'] = self.time
        return zscores

    def _set_event_attrs(self):
        for stimulus in self.eventdata.columns[1:]:
            self.timestamps[stimulus] = list(
                self.eventdata['Time(s)'].iloc[np.where(
                    self.eventdata[stimulus] == 1)[0]])
            if stimulus != 'Lick':
                self.allstim.extend(self.timestamps[stimulus])
        drylicks = [x for x in self.timestamps['Lick'] if x not in self.allstim]
        self.numlicks = len(self.timestamps['Lick'])
        for stim, tslist in self.timestamps.items():
            if stim != 'Lick' and stim != 'Rinse' and len(self.timestamps[stim]) > 0:
                # Get list of tastant deliveries
                tslist.sort()
                tslist = np.array(tslist)
                times = [tslist[0]]  # First one is a trial
                # Delete duplicates
                for i in range(0, len(tslist) - 1):
                    if tslist[i] == tslist[i + 1]:
                        nxt = tslist[i + 1]
                        tslist = np.delete(tslist, i + 1)
                        print(f"Deleted timestamp {nxt}")
                # Add each tastant delivery preceded by a drylick (trial start).
                for ts in tslist:
                    last_stimtime = tslist[np.where(
                        tslist == ts)[0] - 1]
                    last_drytime = drylicks[np.where(
                        drylicks < ts)[0][-1]]
                    if last_drytime > last_stimtime:
                        times.append(ts)
                self.trial_times[stim] = times
        self.drylicks = func.get_matched_time(self.time, drylicks)

    def _set_trace_signals(self) -> pd.DataFrame:

        temp = self.tracedata.copy()
        temp.pop('Time(s)')
        self.signals = temp
        return temp

    def _authenticate(self):

        if not isinstance(self.tracedata, pd.DataFrame):
            raise e.DataFrameError('Trace data must be a dataframe')
        if not isinstance(self.eventdata, pd.DataFrame):
            raise e.DataFrameError('Event data must be a dataframe.')
        if not any(x in self.tracedata.columns for x in ['C0', 'C00']):
            raise AttributeError("No cells found in DataFrame")

    def _get_data(self):
        
        try:
            traces = [file for file in self.filehandler.get_tracedata()][0]
        except IndexError:
            raise e.FileError('File type not located in directory.')
        events = [file for file in self.filehandler.get_eventdata()][0]
        self.tracedata = self._clean(traces)
        self.tracedata['Time(s)'] = np.round(self.tracedata['Time(s)'], 1)
        self.eventdata = events

    @staticmethod
    def _clean(_df) -> pd.DataFrame:

        # When CNMFe is run, we choose cells to "accept", but with manual ROI's every cell is accepted
        # so often that step is skipped. Need some way to check (check_if_accepted)

        def check_if_accepted(_df):
            # If any cells marked as "accepted", use only those cells
            accepted_col = [col for col in _df.columns if ' accepted' in col]
            return accepted_col

        accept = check_if_accepted(_df)

        if accept:
            accepted = np.where(_df.loc[0, :] == ' accepted')[0]
            _df = _df.iloc[:, np.insert(accepted, 0, 0)]
        _df = _df.drop(0)
        _df = _df.rename(columns={' ': 'Time(s)'})
        _df = _df.astype(float)
        _df = _df.reset_index(drop=True)
        _df.columns = [column.replace(' ', '') for column in _df.columns]
        return _df
