#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#taste_data
"""
import pandas as pd
import core.calciumdata as ca
import numpy as np
from utils import funcs as func
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

class TasteData(object):
    def __init__(self, 
                 data,
                 baseline: int = 0,
                 post: int = 0) -> None: 
        
        assert isinstance(data, ca.CalciumData)
        
        self.color_dict = data.color_dict
        self.timestamps = data.timestamps
        self.trial_times = data.trial_times
        
        self.tracedata = data.tracedata
        self.time = None
        self.colors = None
        self.signals = None
        
        self.post = None
        self.baseline = None
        
        self.eventdata = None
        self.tastedata = {}

        self.process(baseline, post)

        self.tastant_dict =  {k: data.color_dict[k] for k in list(data.color_dict)[:6]}
        self.tastants = list(self.tastant_dict.keys())
        

                
    def process(self, baseline: int=0, post: int=0):
        
        logging.info('Setting taste data')
        new_df = pd.DataFrame()
        for event, interv in func.iter_events(
                self.timestamps):
            
            df = self.tracedata.loc[
                (self.tracedata['Time(s)'] > (interv[0]-baseline)) &
                (self.tracedata['Time(s)'] < (interv[1]+post))].copy()
            
            df['colors'] = self.color_dict[event]
            new_df = pd.concat([new_df, df], axis=0)
        new_df.sort_values(by='Time(s)')
        self.baseline = baseline
        self.post = post

        return self._set_data(new_df)
    
    
    def _set_data(self, df):
        
        self.eventdata = df.copy()
        self.time = df.pop('Time(s)')
        self.colors = df.pop('colors')
        self.signals = df
        logging.info('Time, colors, signals are set.')
        self._split()
        
        return None
    
    
    def _split(self):
        for color in np.unique(self.colors):
            event = [tastant for tastant, col in self.color_dict.items() if col in [color]][0]
            self.tastedata[event] = self.eventdata.loc[self.eventdata['colors'] == color]

    
    
    
    
        
