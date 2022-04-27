#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#taste_data
"""
import pandas as pd
from core.calciumdata import CalciumData as ca
import numpy as np
from core import funcs as func
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

class TasteData(object):
    def __init__(self, 
                 data: pd.DataFrame,
                 stamps: dict,
                 color_dict: dict,
                 baseline: int = 0,
                 post: int = 0) -> None: 
        
        
        self.data = data
        self.color_dict = color_dict

        self.time = None
        self.colors = None
        self.signals = None
        self.events: pd.Series
        
        self.post = post
        self.baseline = baseline
        
        self.eventdata = None
        self.tastedata = {}

        self.tastant_dict =  {k: color_dict[k] for k in list(color_dict)[:6]}
        self.tastants = list(self.tastant_dict.keys())
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
            
            df['colors'] = self.color_dict[event]
            df['events'] = event
            new_df = pd.concat([new_df, df], axis=0)
        new_df.sort_values(by='Time(s)')

        return self._set_data(new_df)
    
    
    def _set_data(self, df):
        
        self.eventdata = df.copy()
        self.time = df.pop('Time(s)')
        self.events = df.pop('events')
        self.colors = df.pop('colors')
        self.signals = df
        logging.info('Time, colors, signals are set.')
        self._split()
        
        return None
    
    
    def _split(self):
        for color in np.unique(self.colors):
            event = [tastant for tastant, col in self.color_dict.items() if col in [color]][0]
            self.tastedata[event] = self.eventdata.loc[self.eventdata['colors'] == color]

    
    
    
    
        
