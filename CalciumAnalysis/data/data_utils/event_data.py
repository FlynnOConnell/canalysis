#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# eventdata.py
Module (data.data_utils): Process event/gpio data exported from _video_gpio.isxd file.
"""
from __future__ import annotations

from typing import Sized
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from file_handling.file_handler import FileHandler


@dataclass(slots=True, order=False)
class EventData:
    timestamps: dict = field(init=False)
    trial_times: dict = field(init=False)
    numlicks: Sized | int = field(init=False)
    drylicks: list = field(init=False)
    filehandler: FileHandler = FileHandler

    def __post_init__(self):
        self._set_event_attrs()

    def __len__(self):
        return len(self.numlicks)

    def _set_event_attrs(self):
        allstim = []
        data: pd.DataFrame = self.filehandler.get_eventdata()
        events = data.rename(columns={'Time(s)': 'time'})
        for stimulus in events.columns[1:]:
            self.timestamps[stimulus] = list(
                events['time'].iloc[np.where(
                    events[stimulus] == 1)[0]])
            if stimulus != 'Lick':
                allstim.extend(self.timestamps[stimulus])
        self.drylicks = [x for x in self.timestamps['Lick'] if x not in allstim]
        self.numlicks: Sized | int = len(self.timestamps['Lick'])
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
        return None

    def get_trials(self) -> None:
        """Print number of instances of each event"""
        for stim, trials in self.trial_times.items():
            logging.info(f'{stim} - {len(trials)}')
        return None


# %%

def main():
    datadir = '/Users/flynnoconnell/Documents/Work/Data'
    animal = 'PGT13'
    date = '121021'
    handler = FileHandler(datadir, animal, date)
    eventdata: EventData = EventData(handler)
    return eventdata


if __name__ == "__main__":
    eventdata = main()
