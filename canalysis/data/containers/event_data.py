#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# eventdata.py
Module (data.data_utils): Process event/gpio data exported from _video_gpio.isxd file.
"""
from __future__ import annotations

from typing import Sized, Iterable, Any
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from numpy import ndarray

from canalysis.data.data_utils.file_handler import FileHandler
from canalysis.helpers import funcs


@dataclass(order=False)
class EventData:
    filehandler: FileHandler
    color_dict: dict
    tracedata_time: np.ndarray
    timestamps: field = field(init=False, default_factory=dict)
    trial_times: field = field(init=False, default_factory=dict)
    # Initialize empty placeholders to fill later
    numlicks: Sized | int = field(default_factory=list)
    drylicks: list = field(default_factory=list)
    __allstim: list = field(default_factory=list)
    alltastestim: list = field(default_factory=list)
    nonreinforced: Iterable = field(default_factory=list)
    matched: bool = False

    def __post_init__(self, ):
        self.timestamps: dict = self.__get_timestamps()
        self.drylicks = [x for x in self.timestamps["Lick"] if x not in self.__allstim]
        self.trial_times: dict = self.__get_trial_times()
        self.nonreinforced: ndarray = self.__get_nonreinforced()

    def __repr__(self):
        return type(self).__name__

    def __len__(self, ):
        return len(self.numlicks)

    def __get_timestamps(self, ):
        timestamps: dict = {}
        data: pd.DataFrame = self.filehandler.get_eventdata()
        events = data.rename(columns={"Time(s)": "time"})
        events['time'] = funcs.get_matched_time(self.tracedata_time, events['time'])
        for stimulus in events.columns[1:]:
            timestamps[stimulus] = list(
                events["time"].iloc[np.where(events[stimulus] == 1)[0]]
            )
            if stimulus != "Lick":
                self.__allstim.extend(timestamps[stimulus])
            if stimulus != "Lick" and stimulus != "ArtSal":
                self.alltastestim.extend(timestamps[stimulus])
        self.numlicks: Sized | int = len(timestamps["Lick"])
        self.alltastestim.sort()
        return timestamps

    def __get_nonreinforced(self, ):
        times = []
        lickstamps = np.array(self.timestamps["Lick"])
        intervals = funcs.interval(self.alltastestim, 2)
        for iteration, interv in enumerate(intervals):
            times.extend(
                lickstamps[
                    np.where((lickstamps >= interv[0]) & (lickstamps <= interv[1]))
                ]
            )
        nr = np.setdiff1d(lickstamps, times)
        return nr

    def __get_trial_times(self) -> dict:
        trial_times: dict = {}
        for stim, tslist in self.timestamps.items():
            if stim != "Lick" and stim != "ArtSal" and len(self.timestamps[stim]) > 0:
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
                    last_stimtime = tslist[np.where(tslist == ts)[0] - 1]
                    last_drytime = self.drylicks[np.where(self.drylicks < ts)[0][-1]]
                    if last_drytime > last_stimtime:
                        times.append(ts)
                trial_times[stim] = times
        return trial_times

    def get_trials(self) -> None:
        """Print number of instances of each event"""
        for stim, trials in self.trial_times.items():
            logging.info(f"{stim} - {len(trials)}")
        return None
