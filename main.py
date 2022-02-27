# -*- coding: utf-8 -*-

"""
@author: Dr. Patricia Di Lorenzo Laboratory @ Binghamton University
"""
import pandas as pd
import logging
import data_utils as du
import models as m
import Func as func
pd.set_option('chained_assignment', None)

logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')

# %% Initialize data

animal_id = 'PGT13'
date = '120221'
target_date = '121021'

session = animal_id + '_' + date

tr_cells = ['C00', 'C01', 'C02', 'C03','C04','C05','C06','C09','C10']


#%% Main 

if __name__ == "__main__":
    
    data = du.Data(animal_id, date, tr_cells)
    data_test = du.Data(animal_id, target_date, tr_cells)
    
    nn = m.NeuralNetwork(data, data_test)
    nn.SVM()
    report = nn.get_scores().report
    rep = report.report
    
    

# data.graph_stim()
    
    



    
    
