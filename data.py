#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 02:16:25 2022

@author: flynnoconnell
"""
import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import logging
import warnings
import Func as func
import Plots as plot
from typing import Tuple
import excepts as e

logger = logging.getLogger(__name__)

#%% 

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
    
    def __init__(self, animal, date, tr_cells):
        
        # Session 
        self.animal = animal
        self.date = date
        self.session = animal + '_' + date
        
        # Core
        self.tracedata = Traces(self._get_data(trace=True))
        self.eventdata = Events(self._get_data())
                        
        self.numlicks = self.eventdata.numlicks

        ## Taste Data
        self.taste_trials = None
        self.tr_colors = None
        self.colors_dict = tastant_colors_dict

        self.get_taste_trials(NN=True)
        
        self.tastants = func._get_unique(
            self.taste_trials.tastant)
        self.tr_data = self.taste_trials.filter(tr_cells)
        self.tr_cells = self.tr_data.columns
        self.taste_signals = None
        
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
            
        
    def get_taste_trials(self, NN: bool = False) -> Tuple[pd.DataFrame, np.ndarray, list]:
        
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
                
                holder = pd.DataFrame(columns=taste_signals.columns)
                holder = (self.tracedata.traces.iloc[sig.index])
                holder['tastant'] = event
                taste_signals = pd.concat([taste_signals, holder])
                
                hol = pd.DataFrame(columns=taste_signals.columns)
                hol = (self.tracedata.traces.iloc[sig.index])
                hol['col'] = colors_dict[event]
                color_signals = pd.concat([color_signals, hol])
                
        taste_signals.sort_index(inplace=True)
        self.taste_trials = taste_signals
        self.tr_colors = color_signals['col']

        if func.has_duplicates(self.taste_trials['Time(s)']):
            self.taste_trials.drop_duplicates(subset=['Time(s)'], inplace=True)
            if not self.taste_trials['Time(s)'].is_unique:
                e.DataFrameError('Duplicate values found and not caught.')
        
                
    def plot_stim(self):
        plot.plot_stim(len(self.tracedata.cells),
                       self.tracedata.traces,
                       self.tracedata.time,
                       self.eventdata.timestamps,
                       self.eventdata.trial_times,
                       self.session,
                       colors_dict)
        
        
    def plot_session(self):
        plot.plot_session(self.tracedata.cells,
                          self.tracedata,
                          self.tracedata.time,
                          self.session,
                          self.numlicks,
                          self.eventdata.timestamps)


class Traces(Data):
    
    def __init__(self, df):

        self._authenticate_input_data(df)
        
        self.traces = df
        self.cells = None
        self.time = None
        self.signals = None
        self.binsize = None

        self.result = None
        self.tr_cells = None
        self._pipeline = None
        self._clean()
        self._set_attr()

    @staticmethod
    def _authenticate_input_data(obj):
        # verify input
        if not any(x in obj.columns for x in [' C0', ' C00']):
            raise AttributeError("No cells found in DataFrame")
        elif 'undecided' in obj.columns:
            warnings.warn('DataFrame contains undecided cells, double check that this was intentional.')

    def _set_attr(self) -> None:

        self.cells = self.traces.columns[1:]
        self.time = self.traces['Time(s)']
        self.binsize = self.time[2] - self.time[1]
        
        temp = self.traces.copy()
        temp.pop('Time(s)')
        self.signals = temp
        
    def _clean(self) -> None:

        accepted = np.where(self.traces.loc[0, :] == ' accepted')[0]
        self.traces = self.traces.iloc[:, np.insert(accepted, 0, 0)]
        self.traces = self.traces.drop(0)
        self.traces = self.traces.rename(columns={' ': 'Time(s)'})
        self.traces = self.traces.astype(float)
        self.traces = self.traces.reset_index(drop=True)
        self.traces.columns = [column.replace(' ', '') for column in self.traces.columns]
    
            
class Events(Data):

    def __init__(self, df):
        
        self._authenticate_input_data(df)
        
        self.events = df
        self.evtime = None
        self.timestamps = {}
        self.trial_times = {}
        self.drylicks = []
        self.numlicks: int = None

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

    def _getdata(self):
        return self.df

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
        
                
        
        

