#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#taste_data
"""
import logging

import numpy as np
import pandas as pd
from data.data_utils import funcs as func
from data.calciumdata import CalciumData

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')


class TasteData(object):
    def __init__(self, 
                 data: pd.DataFrame,
                 stamps: dict,
                 baseline: int = 0,
                 post: int = 0) -> None: 
        self.data = data
        self.signals = None
        self.time = None
        self.colors = None
        self.events: None
        self.post = post
        self.baseline = baseline
        self.taste_events = {}
        self.process(stamps)
        

    @staticmethod
    def _authenticate(data):
        assert isinstance(data, pd.DataFrame) and data['Time(s)'] in data.columns         
            
                
    def process(self, stamps):
        logging.info('Setting taste data')
        new_df = pd.DataFrame()
        for event, interv in func.iter_events(stamps):
            df = self.data.loc[
                (self.data['Time(s)'] > (interv[0] - self.baseline)) &
                (self.data['Time(s)'] < (interv[1] + self.post))].copy()
            df['colors'] = CalciumData.color_dict[event]
            df['events'] = event
            new_df = pd.concat([new_df, df], axis=0)
        new_df.sort_values(by='Time(s)')
        return self._set_data(new_df)
    
    
    def _set_data(self, df):
        eventdata = df.copy()
        self.time = df.pop('Time(s)')
        self.events = df.pop('events')
        self.colors = df.pop('colors')
        self.signals = df
        return self._split(eventdata)

    
    def _split(self, eventdata):
        color_dict = CalciumData.color_dict
        for color in np.unique(self.colors):
            event = [tastant for tastant, col in color_dict.items() if col in [color]][0]
            self.taste_events[event] = eventdata.loc[eventdata['colors'] == color]
            
    
    def get_events(self, lst):
        big_df = pd.DataFrame()
        for event, df in self.taste_events.items():
            if event in lst: 
                big_df = pd.concat([big_df, df], axis=0)
        return big_df
                
            
            
        
            
    
            
            
    
    
    
    
    
        
