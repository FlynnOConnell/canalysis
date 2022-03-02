#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#data.py

Module: Classes for data processing.
"""
from __future__ import annotations
from typing import Type, Optional

import os
import pandas as pd
import numpy as np
import logging
import warnings

from utils import funcs as func
from utils import draw_plots as plot
from utils import excepts as e

logger = logging.getLogger(__name__)

# %%

colors_dict = {
    'ArtSal': 'dodgerblue',
    'MSG': 'darkorange',
    'NaCl': 'lime',
    'Sucrose': 'magenta',
    'Citric': 'yellow',
    'Quinine': 'red',
    'Rinse': 'lightsteelblue',
    'Lick': 'darkgray'
}

tastant_colors_dict = {k: colors_dict[k] for k in list(colors_dict)[:]}


class Data(object):

    def __init__(self, animal, date, tr_cells: Optional[any] = ''):

        # Session 
        self.animal = animal
        self.date = date
        self.session = animal + '_' + date
        self.tastant_colors_dict = tastant_colors_dict
        # Core
        self.tracedata = Traces(self._get_data(trace=True), animal, date)
        self.eventdata = Events(self._get_data(), animal, date)

        self.numlicks = self.eventdata.numlicks

        ## Taste Data
        if tr_cells:
            self.all_taste_trials: Type[pd.DataFrame] = pd.DataFrame  # All data for taste-trials
            self.tr_colors = None  # Array of which color of tastant was presented
            self.get_taste_trials()

            self.tastants = func.get_unique(  # List of all tastants (not rinse or lick)
                self.all_taste_trials.tastant)
            self.tr_data = self.all_taste_trials.filter(tr_cells)  # Data for taste-responsive cells only
            self.tr_cells = self.tr_data.columns

        logging.info('Data instantiated.')

    def _get_data(self, trace=False):

        if os.name == 'posix':
            datadir = '/Users/flynnoconnell/Documents/Work/Data'
        else:
            datadir = 'A:\\'

        traces, events, _ = func.get_dir(
            datadir, self.animal, self.date)

        tracedata = traces
        eventdata = events

        if trace:
            return tracedata
        else:
            return eventdata

    def get_taste_trials(self) -> None:

        # Get timestamps of taste - trials only
        stamps = self.eventdata.timestamps.copy()
        stamps.pop('Lick')
        stamps.pop('Rinse')

        taste_signals = pd.DataFrame(columns=self.tracedata.signals.columns)
        color_signals = pd.DataFrame(columns=self.tracedata.signals.columns)

        # Populate df with taste trials only
        for event, timestamp in stamps.items():
            taste_interval = func.interval(timestamp)
            for lst in taste_interval:
                sig_time = func.get_matched_time(self.tracedata.time, *lst)

                sig = (self.tracedata.traces.loc[(
                                                         self.tracedata.traces['Time(s)'] >= sig_time[0])
                                                 & (self.tracedata.traces['Time(s)'] <= sig_time[1]), 'Time(s)'])

                holder = (self.tracedata.traces.iloc[sig.index])
                holder['tastant'] = event
                taste_signals = pd.concat([taste_signals, holder])

                pd.DataFrame(columns=taste_signals.columns)
                hol = (self.tracedata.traces.iloc[sig.index])
                hol['col'] = tastant_colors_dict[event]
                color_signals = pd.concat([color_signals, hol])

        taste_signals.sort_index(inplace=True)
        self.all_taste_trials = taste_signals
        self.tr_colors = color_signals['col']

        if func.has_duplicates(self.all_taste_trials['Time(s)']):
            self.all_taste_trials.drop_duplicates(subset=['Time(s)'], inplace=True)
            if not self.all_taste_trials['Time(s)'].is_unique:
                e.DataFrameError('Duplicate values found and not caught.')

    def plot_stim(self):
        plot.plot_stim(len(self.tracedata.cells),
                       self.tracedata.traces,
                       self.tracedata.time,
                       self.eventdata.timestamps,
                       self.eventdata.trial_times,
                       self.session,
                       tastant_colors_dict)

    def plot_session(self):
        plot.plot_session(self.tracedata.cells,
                          self.tracedata.traces,
                          self.tracedata.time,
                          self.session,
                          self.numlicks,
                          self.eventdata.timestamps)


class Traces(Data):

    def __init__(self, df, animal, date):

        super().__init__(animal, date)
        self._authenticate_input_data(df)

        self.traces: pd.DataFrame = df
        self.cells = self.traces.columns[1:]
        self.time = self.traces['Time(s)']
        self.binsize = self.time[2] - self.time[1]
        self.signals = self._set_signals()

        self.result = None
        self.tr_cells = None
        self._pipeline = None
        self._clean()

    @staticmethod
    def _authenticate_input_data(obj):
        # verify input
        if not any(x in obj.columns for x in [' C0', ' C00']):
            raise AttributeError("No cells found in DataFrame")
        elif 'undecided' in obj.columns:
            warnings.warn('DataFrame contains undecided cells, double check that this was intentional.')

    def _set_signals(self) -> pd.DataFrame:

        temp = self.traces.copy()
        temp.pop('Time(s)')
        self.signals = temp
        return temp

    def _clean(self) -> None:

        accepted = np.where(self.traces.loc[0, :] == ' accepted')[0]
        self.traces = self.traces.iloc[:, np.insert(accepted, 0, 0)]
        self.traces = self.traces.drop(0)
        self.traces = self.traces.rename(columns={' ': 'Time(s)'})
        self.traces = self.traces.astype(float)
        self.traces = self.traces.reset_index(drop=True)
        self.traces.columns = [column.replace(' ', '') for column in self.traces.columns]


class Events(Data):

    def __init__(self, df, animal, date):

        super().__init__(animal, date)
        self._authenticate_input_data(df)

        self.events = df
        self.evtime = None
        self.timestamps = {}
        self.trial_times = {}
        self.drylicks = []
        self.numlicks: int | None = None

        self._set_attr()
        self._validate_attr()

    @staticmethod
    def _authenticate_input_data(obj):
        # verify input
        if not hasattr(obj, "iloc"):
            raise AttributeError('Input data must be a pandas DataFrame')
        if 'Lick' not in obj.columns:
            raise AttributeError("No cells found in DataFrame")

    def _validate_attr(self):
        if not isinstance(self.events, pd.DataFrame):
            raise AttributeError('Event attributes not filled.')

    def _set_attr(self):

        allstim = []
        for stimulus in self.events.columns[1:]:
            self.timestamps[stimulus] = list(
                self.events['Time(s)'].iloc[np.where(
                    self.events[stimulus] == 1)[0]])
            if stimulus != 'Lick':
                allstim.extend(self.timestamps[stimulus])
        self.drylicks = [x for x in self.timestamps['Lick'] if x not in allstim]
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
                    last_drytime = self.drylicks[np.where(
                        self.drylicks < ts)[0][-1]]
                    if last_drytime > last_stimtime:
                        times.append(ts)
                self.trial_times[stim] = times
