# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import logging
import numpy as np

from core.calciumdata import CalciumData
from core.taste_data import TasteData
from graphs.utils.quick_plots import Quick_Plot as qp
from graphs.plot import Plot
from stats.stats import ProcessData 
from core.utils import funcs as func
from models import SVM

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)



# %% Initialize data

datadir = '/Users/flynnoconnell/Documents/Work/Data'
# datadir = 'A://'
animal = 'PGT13'
date = '121021'
data = CalciumData(animal, date, datadir)

    

#%% Other Data

zscores = data.zscores
taste_data_z = TasteData(zscores, data.timestamps, data.color_dict)
taste_data_z.events

# events = func.convert_tastants(taste_data_z.colors, taste_data_z.color_dict)





if __name__ == "__main__":
    pass
