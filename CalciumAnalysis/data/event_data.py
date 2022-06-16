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

from data.data_utils.file_handler import FileHandler


@dataclass(order=False)
class EventData:
    filehandler: FileHandler
    color_dict: dict
    timestamps: field = field(init=False, default_factory=dict)
    trial_times: field = field(init=False, default_factory=dict)
    # Initialize empty placeholders to fill later
    numlicks: Sized | int = field(default_factory=list)
    drylicks: list = field(default_factory=list)
    __allstim: list = field(default_factory=list)

    def __post_init__(self):
        self.timestamps: dict = self.__get_timestamps()
        self.drylicks = [x for x in self.timestamps['Lick'] if x not in self.__allstim]
        self.trial_times: dict = self.__get_trial_times()


    def __len__(self):
        return len(self.numlicks)

    def __get_timestamps(self):
        timestamps: dict = {}
        data: pd.DataFrame = self.filehandler.get_eventdata()
        events = data.rename(columns={'Time(s)': 'time'})
        for stimulus in events.columns[1:]:
            timestamps[stimulus] = list(
                events['time'].iloc[np.where(
                    events[stimulus] == 1)[0]])
            if stimulus != 'Lick':
                self.__allstim.extend(timestamps[stimulus])
        self.numlicks: Sized | int = len(timestamps['Lick'])
        return timestamps

    def __get_trial_times(self) -> dict:
        trial_times: dict = {}
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
        return trial_times

    def get_trials(self) -> None:
        """Print number of instances of each event"""
        for stim, trials in self.trial_times.items():
            logging.info(f'{stim} - {len(trials)}')
        return None
