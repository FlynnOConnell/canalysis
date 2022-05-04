#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# tracedata.py
Module (data.data_utils): Process traces exported from inscopix trace file.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats

from data.data_utils.file_handler import FileHandler

#%%

@dataclass
class TraceData: 
    filehandler: FileHandler = FileHandler
    def __post_init__(self):
        self.tracedata = next(self.filehandler.get_tracedata())
        self._clean()
        #Core attributes
        self.signals = self._set_trace_signals()
        self.cells = np.array(self.tracedata.columns[1:])
        self.time = self.tracedata.time
        self.binsize = self.time[2] - self.time[1]
        self.zscores = self._get_zscores()
    
    
    def __hash__(self):
        return hash(repr(self))


    def __len__(self):
        return len(self.numlicks)
    
    
    def _get_zscores(self) -> pd.DataFrame:

        zscores = pd.DataFrame(columns=self.signals.columns)
        for cell in self.signals.columns:
            allsignal = self.signals[cell]
            zscore = pd.Series(stats.zscore(allsignal))
            zscores[cell] = zscore
        zscores['time'] = self.time
        return zscores
    
    
    def _set_trace_signals(self) -> pd.DataFrame:
        temp = self.tracedata.copy()
        temp.pop('time')
        self.signals = temp
        return temp
    
    
    def _clean(self) -> pd.DataFrame:
        
        # When CNMFe is run, we choose cells to "accept", but with manual ROI's every cell is accepted
        # so often that step is skipped. Need some way to check (check_if_accepted)
        _df = self.tracedata.copy()
        def check_if_accepted(_df):
            # If any cells marked as "accepted", use only those cells
            accepted_col = [col for col in _df.columns if ' accepted' in col]
            return accepted_col
        accept = check_if_accepted(_df)
        if accept:
            accepted = np.where(_df.loc[0, :] == ' accepted')[0]
            _df = _df.iloc[:, np.insert(accepted, 0, 0)]
        _df = _df.drop(0)
        _df = _df.rename(columns={' ': 'time'})
        _df = _df.astype(float)
        _df = _df.reset_index(drop=True)
        _df.columns = [column.replace(' ', '') for column in _df.columns]
        _df['time'] = np.round(_df['time'], 1)
        self.tracedata = _df
        return None
    
#%%

def main():
    datadir = '/Users/flynnoconnell/Documents/Work/Data'
    animal = 'PGT13'
    date = '121021'
    handler = FileHandler(datadir, animal, date)
    tracedata = TraceData(handler)
    return tracedata



if __name__ == "__main__":
    tracedata = main()

    
