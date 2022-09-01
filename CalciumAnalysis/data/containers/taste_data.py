#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#taste_data
"""
import logging
from typing import Generator, Any, Iterable
import pandas as pd
from dataclasses import dataclass
from utils import funcs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


@dataclass
class TasteData:
    __signals: pd.DataFrame
    __time: pd.Series
    __timestamps: dict
    color_dict: dict
    baseline: int = 0
    post: int = 4

    def __post_init__(self):
        assert isinstance(self.__signals, pd.DataFrame)
        self.tastedata = self.concat_event_signals(self.__timestamps).drop(columns=['time'])
        self.events: pd.Series = self.tastedata['event']
        self.colors: pd.Series = self.tastedata['color']
        self.signals = self.tastedata.drop(columns=['event', 'color'])

    def __repr__(self):
        return type(self).__name__

    def concat_event_signals(self, timestamps):
        """
        From data.timestamps, iterate each event and get signals
        starting from interval[0] to interval [1].

        Parameters
        ----------
        self : instance
        timestamps : dict
            Timestamps to sort through.
        Returns
        -------
        method
            Call to method where specific attributes are set.
        """
        logging.info("Setting taste data...")
        aggregate_signals_df = pd.DataFrame()
        for event, interv in funcs.iter_events(timestamps):
            if event not in ['Lick', 'ArtSal']:
                event_df = self.__signals.loc[
                    (self.__time > (interv[0] - self.baseline)) &
                    (self.__time < (interv[1] + self.post))].copy()
                event_df['color'] = self.color_dict[event]
                event_df["event"] = event
                aggregate_signals_df = pd.concat([
                    aggregate_signals_df, event_df],
                    axis=0)
        logging.info("Taste data set.")
        return aggregate_signals_df

    def get_signals_from_events(self, events: list) -> tuple[pd.DataFrame, pd.Series]:
        signal = self.tastedata[self.tastedata['event'].isin(events)].drop(columns=['event'])
        color = signal.pop('color')
        return signal, color

