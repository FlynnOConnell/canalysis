#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#taste_data
"""
import logging
from collections import namedtuple
from typing import Generator, Any, Iterable
import numpy as np
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
    color_dict: namedtuple
    baseline: int = 0
    post: int = 4

    def __post_init__(self):
        self.taste_events = {}
        self.concat_event_signals(self.__timestamps)
        assert isinstance(self.__signals, pd.DataFrame)

    def generate_stim_data(self) -> Generator[pd.DataFrame, None, None]:
        yield [(stim, df, df['colors']) for stim, df in self.taste_events.items()]

    def _split(self, eventdata):
        """Create dictionary with each event and associated traces"""
        for color in np.unique(eventdata['colors']):
            event = [
                tastant for tastant, col in self.color_dict.items() if col in [color]
            ][0]
            self.taste_events[event] = eventdata.loc[eventdata["colors"] == color]

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
                event_df["colors"] = self.color_dict[event]
                aggregate_signals_df = pd.concat([
                    aggregate_signals_df, event_df],
                    axis=0)
        logging.info("Taste data set.")
        return self._split(aggregate_signals_df)

    def get_events(self, lst):
        """
        Create large dataframe given a list of events.
        Parameters
        ----------
        lst : list
            List of events used to create large DataFrame.\
        Returns
        -------
        big_df : pd.DataFrame
            Concatenated dataframe with each event.
        """
        new_df = pd.DataFrame()
        for event, df in self.taste_events.items():
            if event in lst:
                new_df = pd.concat([new_df, df], axis=0)
        return new_df
