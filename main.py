# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import os
import pandas as pd
import numpy as np
import logging

import data as du
import models as m
from utils import funcs as func
from sklearn import preprocessing

# %% Initialize data

pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')


# datadir = '/Users/flynnoconnell/Documents/Work/Data'

datadir = 'A:\\'
animal_id = 'PGT13'

date = '011222'
target_date = '121021'

tr_cells = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C09', 'C10']

#%%

if __name__ == "__main__":
    
    data = du.Data(animal_id, date, datadir, tr_cells)
    data_test = du.Data(animal_id, target_date, datadir, tr_cells)
    nn = m.NeuralNetwork(data, data_test)

    nn.SVM()

    # x =  nn.target
    # xx = nn.target_enc

    # x_enc = nn.target
    # nn.SVM()
    # report = nn.get_scores().report
    # rep = report.report
