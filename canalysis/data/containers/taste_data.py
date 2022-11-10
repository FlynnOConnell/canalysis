#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#taste_data
"""
import logging
from typing import Generator, Iterable, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from canalysis.helpers import funcs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


@dataclass
class TasteData:
    __signals: pd.DataFrame
    __time: pd.Series
    __timestamps: dict
    trial_times: dict
    __avgs: dict
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

    def get_signals_from_events(self, events: list) -> Tuple[pd.DataFrame, pd.Series]:
        signal = self.tastedata[self.tastedata['event'].isin(events)].drop(columns=['event'])
        color = signal.pop('color')
        return signal, color

    def get_taste_df(
            self, zero: bool=True,
    ) -> Generator[Iterable, None, None]:
        signals = self.__signals.drop("time", axis=1)
        if zero:
            for cell in signals:
                signals[cell] = signals[cell] - self.__avgs[cell]
                # Replace negatives with 0 using numpys fancy indexing
                signals[cell][signals[cell] < 0] = 0

        for stim, times in self.trial_times.items():
            for iteration, trial in enumerate(times):
                data_ind = np.where((self.__time > trial-2) & (self.__time < trial + 5))[0]
                signal = signals.iloc[data_ind, :]
                yield stim, iteration, signal

    def loop_taste(
            self,
            save_dir: Optional[str] = "",
            **kwargs
    ) -> Generator[Iterable, None, None]:
        from heatmaps import EatingHeatmap
        for stim, iteration, signal in self.get_taste_df():
            heatmap = EatingHeatmap(
                    signal.T,
                    title=f'{stim}, Trial: {iteration + 1}',
                    save_dir=save_dir,
                    save_name=f'{stim}_{iteration}',
                    **kwargs
            )
            fig = heatmap.default_heatmap(maptype='taste')
            yield fig
