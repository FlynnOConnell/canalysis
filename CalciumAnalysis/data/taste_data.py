#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#taste_data
"""
import logging

import numpy as np
import pandas as pd
from data.data_utils import funcs as func

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')


class TasteData(object):
    def __init__(self, 
                 data: pd.DataFrame,
                 stamps: dict,
                 color_dict,
                 baseline: int = 0,
                 post: int = 0) -> None: 
        """
        Subset of CalciumData, with specific attributes based on events.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with cells and time included.
        stamps : dict
            Event timestamps.
        color_dict : dict
            Event : color mappings.
        baseline : int, optional
            Number of seconds to include in baseline. The default is 0.
        post : int, optional
            Number of seconds to include post: baseline. The default is 5.

        Returns
        -------
        None
            DESCRIPTION.

        """
        self.data = data
        self.signals = None
        self.time = None
        self.colors = None
        self.events: None
        self.post = post
        self.baseline = baseline
        self.color_dict = color_dict
        self.taste_events = {}
        self.process(stamps)
        
      
    def get_binary(self, class_0, class_1):
        """
        Get binary classification labels (+1 or -1) for neural network.
    
        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            Data used for classification, without labels.
        labels_0 : pd.Series | np.ndarray
            Labels to include in +1 label.
        labels_1 : pd.Series | np.ndarray
            Labels to include in -1 label.
        Returns
        -------
        None.
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
            
            
    @staticmethod
    def _authenticate(data):
        """Type check input data"""
        assert isinstance(data, pd.DataFrame) and data['time'] in data.columns         
            
                
    def process(self, stamps):
        """
        Get taste_events from all events with a given
        baseline and post-baseline period.

        Parameters
        ----------
        stamps : dict
            Timestamps to sort through.

        Returns
        -------
        method
            Call to method where specific attributes are set.

        """
        logging.info('Setting taste data')
        new_df = pd.DataFrame()
        for event, interv in func.iter_events(stamps):
            df = self.data.loc[
                (self.data['time'] > (interv[0] - self.baseline)) &
                (self.data['time'] < (interv[1] + self.post))].copy()
            df['colors'] = self.color_dict[event]
            df['events'] = event
            new_df = pd.concat([new_df, df], axis=0)
        new_df.sort_values(by='time')
        return self._set_data(new_df)
    
    
    def _set_data(self, df):
        """Set attributes from processed CalciumData"""
        eventdata = df.copy()
        self.time = df.pop('time')
        self.events = df.pop('events')
        self.colors = df.pop('colors')
        self.signals = df
        return self._split(eventdata)

    
    def _split(self, eventdata):
        """Create dictionary with each event and associated traces"""
        for color in np.unique(self.colors):
            event = [tastant for tastant, col in self.color_dict.items() if col in [color]][0]
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
    
        
                
            
            
        
            
    
            
            
    
    
    
    
    
        
