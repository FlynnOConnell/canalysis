#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#taste_data
"""
import logging
from collections import namedtuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from misc import funcs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')


@dataclass
class TasteData:
    signals: pd.DataFrame
    time: pd.Series
    timestamps: dict
    color_dict: namedtuple
    baseline: int = 0
    post: int = 0

    def __post_init__(self):
        self.taste_events = {}
        self.process(self.timestamps)
        self._authenticate()

    def get_binary(self, class_0, class_1):
        """
        Get binary classification labels (+1 or -1) for neural network.
        """
        df_0 = pd.DataFrame(columns=self.signals.columns)
        df_1 = pd.DataFrame()
        for key, df in self.taste_events.items():
            df = df.drop(['time', 'colors'], axis=1)
            if key in class_0:
                df['binary'] = '0'
                df_0 = pd.concat([df_0, df], axis=0)
            if key in class_1:
                df['binary'] = '1'
                df_1 = pd.concat([df_1, df], axis=0)
        return pd.concat([df_0, df_1], axis=0)

    def _authenticate(self):
        """Type-check input data"""
        assert isinstance(self.signals, pd.DataFrame)

    def process(self, timestamps):
        """
        Get taste_events from all events with a given
        baseline and post-baseline period.

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
        logging.info('Setting taste data')
        new_df = pd.DataFrame()
        for event, interv in funcs.iter_events(timestamps):
            df = self.signals.loc[
                (self.time > (interv[0] - self.baseline)) &
                (self.time < (interv[1] + self.post))].copy()
            df['time'] = self.time
            df['colors'] = self.color_dict.__getattribute__(event)
            df['events'] = event
            new_df = pd.concat([new_df, df], axis=0)
        new_df.sort_values(by='time')
        return self._set_data(new_df)

    def _set_data(self, df):
        """Set new attributes from processed CalciumData"""
        eventdata = df.copy()
        self.time = df.pop('time')
        self.events = df.pop('events')
        self.colors = df.pop('colors')
        self.signals = df
        return self._split(eventdata)

    def _split(self, eventdata):
        """Create dictionary with each event and associated traces"""
        for color in np.unique(self.colors):
            event = \
                [tastant for tastant, col in self.color_dict._asdict().items() if col in [color]][0]
            self.taste_events[event] = eventdata.loc[eventdata['colors'] == color]

    def get_events(self, lst):
        """
        Create large dataframe given a list of events.

        Parameters
        ----------
        lst : list
            List of events used to create large DataFrame.

        Returns
        -------
        big_df : TYPE
            DESCRIPTION.

        """
        new_df = pd.DataFrame()
        for event, df in self.taste_events.items():
            if event in lst:
                new_df = pd.concat([new_df, df], axis=0)
        return new_df
