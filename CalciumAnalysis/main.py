# -*- coding: utf-8 -*-

"""
#main_nn.py

Module: Main code execution. 
        Note: Neural network requires separate main.py in neuralnetwork subpackage.
"""
import logging

from data.calciumdata import CalciumData
from data.taste_data import TasteData
from graphs.plot import Plot
from stats.stats import Stats

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)



# %% Some examples on usage of different modules.

def initialize_data():
    """Set directory where data is stored."""
    
    datadir = '/Users/flynnoconnell/Documents/Work/Data'
    animal = 'PGT13'
    date = '121021'
    data = CalciumData(animal, date, datadir)
    return data

def sparese_event_data(data):
    """Get specific subset of data based on particular events"""
    taste_data = TasteData(data.tracedata, data.timestamps, data.color_dict)
    return taste_data



    
if __name__ == "__main__":
    initialize_data()
