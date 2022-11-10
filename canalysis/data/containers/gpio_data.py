# -*- coding: utf-8 -*-
"""
# process_gpio.py

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from canalysis.data.data_utils.file_handler import FileHandler


@dataclass
class GpioData:
    filehandler: FileHandler = FileHandler
    threshold: int = 3000
    dist_adjust: float = 0.005
    decode_gpio = {
        "Rinse": [1, 2, 3, 4],
        "ArtSal": [1, 3, 4],
        "Quinine": [1, 2, 4],
        "NaCl": [1, 3],
        "Sucrose": [1, 2],
        "MSG": [1, 4],
        "Citric": [1, 2, 3],
    }
    gpiodata = None

    def __post_init__(self):
        self.gpiodata = self.filehandler.get_gpiodata()
        self.timestamps = {}

    @staticmethod
    def within(arr, ts, adjust):
        arr = np.array(arr)
        out = arr[np.where((arr > ts - adjust) & (arr < ts + adjust))]
        return out

    def get_rec_end(self):
        rec_end = max(
            self.gpiodata.loc[
                np.where(self.gpiodata[" Channel Name"] == " BNC Sync Output")[0],
                "Time (s)",
            ]
        )
        return rec_end

    def trim(self):
        self.gpiodata = self.gpiodata.iloc[
            np.where(self.gpiodata[" Channel Name"].str.contains("GPIO"))[0], :
        ]

    def get_timestamps(self):
        for chan in pd.unique(self.gpiodata[" Channel Name"]):
            print("Processing events for:", chan)
            event = []
            gp_chan = self.gpiodata.iloc[
                np.where(self.gpiodata[" Channel Name"] == chan)[0], :
            ]
            gp_check = gp_chan[gp_chan[" Value"] > self.threshold]
            for index, row in gp_check.iterrows():
                t = row["Time (s)"]
                # see if any times in the last 11 ms exceed the threshold.
                # pulses last 10 ms, should not any true values
                tcomp = (
                    gp_chan[
                        (gp_chan["Time (s)"] >= t - 0.011) & (gp_chan["Time (s)"] < t)
                    ][" Value"]
                    > self.threshold
                )
                # if all checks are good add the event
                if row[" Value"] > self.threshold and not np.any(tcomp):
                    event.append(t)
            self.timestamps[chan] = event
        self.timestamps["Lick"] = self.timestamps[" GPIO-1"]

    def _clean_validate(self):
        for i in range(2, 5):  # for channels 2 through 4
            for ts in self.timestamps[" GPIO-{}".format(i)]:
                # if the timestamp is not in channel 1 (all licks)
                if ts not in self.timestamps[" GPIO-1"]:
                    # get all nearby timestamps from GPIO1
                    tscheck = self.within(
                        self.timestamps[" GPIO-1"], ts, self.dist_adjust
                    )
                    if len(tscheck) == 1:  # there should only be one
                        # if so replace the timestamp with the one from GPIO-1
                        print(
                            "Adjusted GPIO-{} timestamps from {} to {}".format(
                                i, ts, tscheck[0]
                            )
                        )
                        self.timestamps[" GPIO-{}".format(i)].remove(ts)
                        self.timestamps[" GPIO-{}".format(i)].append(tscheck[0])
            self.timestamps[" GPIO-{}".format(i)].sort()

    def _collect_gpio(self):
        allts = []
        # this loop collects all GPIO timestamps
        for chan in pd.unique(self.gpiodata[" Channel Name"]):
            allts.extend(self.timestamps[chan])

    def decode(self):
        for stim, inputs in self.decode_gpio.items():
            ts_holder = []
            for ts in self.timestamps[" GPIO-1"]:  # for each lick
                for chan in range(1, 5):
                    if (
                        ts not in self.timestamps[" GPIO-{}".format(chan)]
                        and chan in inputs
                    ):
                        break  # if the input is needed to code for this stimulus and isn't present, move on
                    elif (
                        ts in self.timestamps[" GPIO-{}".format(chan)]
                        and chan not in inputs
                    ):
                        break  # if the input is not needed to code for this stimulus and is present, move on
                else:
                    ts_holder.append(ts)
            self.timestamps[stim] = ts_holder
