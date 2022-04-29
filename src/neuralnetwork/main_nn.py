# -*- coding: utf-8 -*-

"""
#main_nn.py
Module: Code execution.
"""
import logging

from data.calciumdata import CalciumData
from data.taste_data import TasteData
from neuralnetwork.SVM import SupportVectorMachine
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# %% Initialize data

datadir = '/Users/flynnoconnell/Documents/Work/Data'
# datadir = 'A://'
animal = 'PGT13'
date = '121021'
data = CalciumData(animal, date, datadir)
td = data.tastedata.taste_events
my_ev = ['Sucrose', 'Quinine', 'NaCl', 'Citric']
my_df = data.tastedata.get_events(my_ev).drop(columns=['colors', 'Time(s)'])
my_df_ev = my_df.pop('events')

model = SVC(C=250, kernel='rbf', gamma='auto', verbose=True)

#%% Other Data

svm = SupportVectorMachine(my_df, my_df_ev)
svm.split()
svm.scale_encode()
# best = svm.optimize_clf()
clf = svm.fit_clf(model=model)


if __name__ == "__main__":
    pass
