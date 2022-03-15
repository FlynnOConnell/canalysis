# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import pandas as pd
import logging

from models.SVM import SupportVectorMachine

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from core.data import CalciumData

pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')

# %% Initialize data

datadir = '/Users/flynnoconnell/Documents/Work/Data'
animal_id = 'PGT13'
date = '121021'
target_date = '011222'

tr_cells = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C09', 'C10']

train_data = CalciumData(animal_id, date, datadir, tr_cells)
val_data = CalciumData(animal_id, target_date, datadir, tr_cells)

cv = StratifiedShuffleSplit()
svm = SupportVectorMachine(train_data, val_data)
clf = svm._get_classifier()

svm.get_learning_curves(clf, cv)

svm.fit(mat=False)

svm.validate()

results = svm.summary
eval_scores = svm.summary['eval_scores']
train_scores = svm.summary['train_scores']


#not sure where to put this yet
def bigloop():
    from models.SVM import run_all_params
    x = svm.X
    y = svm.y
    
    run_all_params(x, y, small=False)

# %%

if __name__ == "__main__":
    pass
    # bigloop()