# -*- coding: utf-8 -*-

"""
#main_nn.py

Module: Code execution.
"""
import logging

from calciumdata import CalciumData
from graphs.plot import Plot
from taste_data import TasteData

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)



# %% Initialize data

datadir = '/Users/flynnoconnell/Documents/Work/Data'
# datadir = 'A://'
animal = 'PGT13'
date = '121021'
data = CalciumData(animal, date, datadir)


    
#%%
zdata = data.zscores

taste_data = TasteData(zdata, data.timestamps, data.color_dict)

zt = zdata.pop('Time(s)')
# %%

# p = qp(zdata, data.time, 'raw')
# p.line_signals('raw')
# p.line_fourier('raw')

# %%
ts = data.timestamps
rem_list = ['ArtSal', 'Citric', 'Lick', 'Rinse']
[ts.pop(key) for key in rem_list]

tastedata = TasteData(data.tracedata, ts, data.color_dict)
t_sig = tastedata.signals
t_col = tastedata.colors

stat = Stats(data)
pca = stat.PCA(t_sig, t_col)

plot = Plot()
plot.scatter(pca[0], colors=t_col)

if __name__ == "__main__":
    pass
