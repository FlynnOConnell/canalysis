#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# eventdata.py
Module (data.data_utils): Process event/gpio data exported from _video_gpio.isxd file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from data.data_utils.file_handler import FileHandler


@dataclass
class EventData: 
    filehandler: FileHandler = FileHandler
    def __post_init__(self,):
        """
        Data container for events. 
        
        Parameters
        ----------
        eventdata: pd.DataFrame
            Frame with timestamps for each event.
        timestamps: dict
            Key ('Event'): Each timestamp for that event 
        trial_times: dict
            Key ('Event'): Timestamp for first trial of each stimulus
        """
        self.eventdata = next(self.filehandler.get_eventdata())
        self.timestamps = {}
        self.trial_times = {}
        self.numlicks: int 
        self._set_event_attrs()
        
    
    def __hash__(self):
        return hash(repr(self))


    def __len__(self):
        return len(self.numlicks)
    
    
    def _set_event_attrs(self):
        allstim = []
        self.eventdata = self.eventdata.rename(columns={'Time(s)': 'time'})
        for stimulus in self.eventdata.columns[1:]:
            self.timestamps[stimulus] = list(
                self.eventdata['time'].iloc[np.where(
                    self.eventdata[stimulus] == 1)[0]])
            if stimulus != 'Lick':
                allstim.extend(self.timestamps[stimulus])
        drylicks = [x for x in self.timestamps['Lick'] if x not in allstim]
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
        return None
        
        
    def get_trials(self) -> None:
        for stim, trials in self.trial_times.items():
            logging.info(f'{stim} - {len(trials)}')
        return None
        
        
#%%

def main():
    datadir = '/Users/flynnoconnell/Documents/Work/Data'
    animal = 'PGT13'
    date = '121021'
    handler = FileHandler(datadir, animal, date)
    eventdata = EventData(handler)
    return eventdata


if __name__ == "__main__":
    eventdata = main()
    