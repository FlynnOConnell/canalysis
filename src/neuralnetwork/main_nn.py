# -*- coding: utf-8 -*-

"""
#main_nn.py
Module: Code execution.
"""
import logging

from calciumdata import CalciumData
from taste_data import TasteData
from SVM import SupportVectorMachine

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
signals = taste_data_z.signals.shape[0]
events = taste_data_z.events.shape[0]
svm = SupportVectorMachine(signals, events)
svm.split()

if __name__ == "__main__":
    pass
