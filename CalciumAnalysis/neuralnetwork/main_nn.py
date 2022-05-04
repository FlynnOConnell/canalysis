# -*- coding: utf-8 -*-

"""
#main_nn.py
Module: Code execution.
"""
import logging

from data.calciumdata import CalciumData

from neuralnetwork.SVM import SupportVectorMachine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# %% Initialize data

datadir = '/Users/flynnoconnell/Documents/Work/Data'
# datadir = 'A://'
animal = 'PGT13'
date = '121021'
data = CalciumData(animal, date, datadir)


#%%

rinse = ['Rinse']
taste = ['Sucrose', 'NaCl', 'Quinine', 'Citric', 'MSG']
df = data.tastedata.get_binary(rinse, taste)
df_labs = df.pop('events')
df_target = df.pop('binary')

#%% Run SVM
svm = SupportVectorMachine(df, df_target, 'train')
svm.split()
svm.scale()
svm.optimize_clf()
svm.fit_clf()
svm.predict_clf()
svm.evaldata = to_pass
svm.evaluate_clf()
evaluation = svm.eval_scoring

taste_svm.cv = cv


#%%

if __name__ == "__main__":
    pass
