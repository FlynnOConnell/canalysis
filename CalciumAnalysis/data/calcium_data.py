#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#data.py

Module: Classes for data processing.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from .data_utils.file_handler import FileHandler
import matplotlib

# matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt

from .all_data import AllData
from .trace_data import TraceData
from .taste_data import TasteData
from .event_data import EventData
from graphs.graph_utils import Mixins
from utils import excepts as e
from utils import funcs

logger = logging.getLogger(__name__)


# %%


@dataclass
class CalciumData(Mixins.CalPlots):
    filehandler: FileHandler
    color_dict: color_dict
    tracedata: TraceData = field(init=False)
    eventdata: EventData = field(init=False)
    _tastedata: TasteData = field(init=False)
    adjust: Optional[int] = None

    alldata: ClassVar[AllData] = AllData.Instance()

    def __post_init__(self):
        self.date = self.filehandler.date
        self.animal = self.filehandler.animal
        self.data_dir = self.filehandler.directory

        # Core
        self.tracedata: TraceData = self._set_tracedata()
        self.eventdata: EventData = self._set_eventdata()
        if self.filehandler.eatingname is not None:
            self.eatingdata = self._set_eatingdata()
            self.eatingsignals = self._set_eating_signals()
        self.nr_avgs = self._get_nonreinforced_signals()

        self._authenticate()

        self._tastedata: TasteData = TasteData(
            self.tracedata.signals,
            self.tracedata.time,
            self.eventdata.timestamps,
            self.color_dict,
        )
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
        return EventData(self.filehandler, self.color_dict)

    def _set_eatingdata(self) -> pd.DataFrame:
        eatingdata = self.filehandler.get_eatingdata()
        eatingdata.drop(
            ["Marker Type", "Marker Event Id", "Marker Event Id 2", "value1", "value2"],
            axis=1,
            inplace=True,
        )
        eatingdata = eatingdata.loc[
            eatingdata["Marker Name"].isin(["Entry", "Eating", "Grooming"])
        ]
        for column in eatingdata.columns[1:]:
            eatingdata[column] = eatingdata[column] + self.adjust

        eatingdata = eatingdata.sort_values("TimeStamp")
        return eatingdata

    def _set_eating_signals(self):
        sort = self.eatingdata.sort_values("TimeStamp")
        return sort

    def _get_nonreinforced_signals(self):
        ev_time = funcs.get_matched_time(
            self.tracedata.time, self.eventdata.nonreinforced
        )
        nr_signal = self.tracedata.tracedata.loc[
            self.tracedata.tracedata["time"].isin(ev_time)
        ].drop("time", axis=1)
        avgs = {}
        for column in nr_signal.columns:
            mean = nr_signal[column].mean()
            avgs[column] = mean
        return avgs

    def _authenticate(self):
        if not isinstance(self.tracedata, TraceData):
            raise e.DataFrameError("Trace data must be a dataframe")
        if not isinstance(self.eventdata, EventData):
            raise e.DataFrameError("Event data must be a dataframe.")
        if not any(
            x in self.tracedata.signals.columns for x in ["C0", "C00", "C000", "C0000"]
        ):
            logging.debug(f"{self.tracedata.signals.head()}")
            raise AttributeError(
                f"No cells found in DataFrame: " f"{self.tracedata.signals.head()}"
            )
        return None

    def _add_instance(self):
        my_dict = type(self).alldata
        if self.keys_exist(my_dict, self.animal, self.date):
            logging.info(f"{self.animal}-{self.date} already exist.")
        elif self.keys_exist(my_dict, self.animal) and not self.keys_exist(
            my_dict, self.date
        ):
            my_dict[self.animal][self.date] = self
            logging.info(f"{self.animal} exists, {self.date} added.")
        elif not self.keys_exist(my_dict, self.animal):
            my_dict[self.animal] = {self.date: self}
            logging.info(f"{self.animal} and {self.date} added")
        return None

    def get_signal(self, i):
        return list(self.tracedata.signals.iloc[:, i])

    def plot_session(
        self, lickshade: Optional[int] = 1, save_dir: Optional[str] = ""
    ) -> None:

        # create a series of plots with a shared x-axis
        fig, axs = plt.subplots(
            len(self.tracedata.cells), 1, sharex=True, facecolor="white"
        )
        for i in range(len(self.tracedata.cells)):
            # get calcium trace (y axis data)
            signal = self.get_signal(i)

            # plot signal
            axs[i].plot(self.tracedata.time, signal, "k", linewidth=0.8)

            # Get rid of borders
            axs[i].get_xaxis().set_visible(False)
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].set_yticks([])  # no Y ticks

            # add the cell name as a label for this graph's y-axis
            axs[i].set_ylabel(
                self.tracedata.signals.columns[i],
                rotation="horizontal",
                labelpad=15,
                y=0.1,
                fontweight="bold",
            )

            # go through each lick and shade it in
            for lick in self.eventdata.timestamps["Lick"]:
                label = "_yarp"
                if lick == self.eventdata.timestamps["Lick"][0]:
                    label = "Licking"
                axs[i].axvspan(
                    lick, lick + lickshade, color="lightsteelblue", lw=0, label=label
                )

            for ind in self.eatingdata.index:
                if self.eatingdata["Marker Name"][ind] == "Entry":
                    color = "blue"
                elif self.eatingdata["Marker Name"][ind] == "Eating":
                    color = "red"
                else:
                    color = "green"
                axs[i].axvspan(
                    self.eatingdata["TimeStamp"][ind],
                    self.eatingdata["TimeStamp2"][ind],
                    alpha=0.5,
                    color=color,
                )

        # Make the plots act like they know each other
        fig.subplots_adjust(hspace=0)
        plt.xlabel("Time (s)")
        fig.suptitle(f"Calcium Traces: {self.filehandler.session}", y=0.95)
        # plt.legend(loc=(1.04, 1))
        axs[-1].get_xaxis().set_visible(True)
        axs[-1].spines["bottom"].set_visible(True)
        logging.info("Figure created.")
        plt.show()

        if save_dir:
            fig.savefig(
                f"{save_dir}/session.png",
                bbox_inches="tight",
                dpi=1000,
                facecolor="white",
            )
            logging.info("Session figure saved.")

        return None

    def plot_zoom(self, zoomshade: float = 0.2, save_dir: Optional[str] = "") -> None:

        # create a series of plots with a shared x-axis
        fig, ax = plt.subplots(len(self.tracedata.cells), 1, sharex=True)
        zoombounding = [
            int(input("Enter start time for zoomed in graph (seconds):")),
            int(input("Enter end time for zoomed in graph (seconds):")),
        ]

        for i in range(len(self.tracedata.cells)):
            signal = self.get_signal(i)

            # plot signal
            ax[i].plot(self.tracedata.time, signal, "k", linewidth=0.8)
            ax[i].get_xaxis().set_visible(False)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].set_yticks([])
            ax[i].set_ylabel(
                self.tracedata.signals.columns[i],
                rotation="horizontal",
                labelpad=15,
                y=0.1,
            )

            # go through each set of timestamps and shade them accordingly
            for stim, times in self.eventdata.timestamps.items():
                for ts in times:
                    if ts == times[0]:
                        label = stim
                    else:
                        label = "_"  # Keeps label from showing.
                    ax[i].axvspan(
                        ts,
                        ts + zoomshade,
                        color=self.color_dict[stim],
                        label=label,
                        lw=0,
                    )

            for ind in self.eatingdata.index:
                if self.eatingdata["Marker Name"][ind] == "Entry":
                    color = "blue"
                elif self.eatingdata["Marker Name"][ind] == "Eating":
                    color = "red"
                else:
                    color = "green"
                ax[i].axvspan(
                    self.eatingdata["TimeStamp"][ind],
                    self.eatingdata["TimeStamp2"][ind],
                    alpha=0.5,
                    color=color,
                )

        # Make the plots act like they know each other
        fig.subplots_adjust(hspace=0)
        plt.xlabel("Time (s)")
        fig.suptitle(f"Calcium Traces: {self.filehandler.session}", y=0.95)
        ax[-1].get_xaxis().set_visible(True)
        ax[-1].spines["bottom"].set_visible(True)
        plt.setp(ax, xlim=zoombounding)
        plt.show()
        if save_dir:
            fig.savefig(
                f"{save_dir}zoom.png", bbox_inches="tight", dpi=1000, facecolor="white"
            )
        return None
