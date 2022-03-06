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
from utils import funcs as func
from utils import data_manipulation as dm
from graphing.draw_plots import Plot
from neural_network import SVM

from sklearn import preprocessing
from sklearn import model_selection


# %% Initialize data

pd.set_option('chained_assignment', None)
logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')

datadir = '/Users/flynnoconnell/Documents/Work/Data'
animal_id = 'PGT13'
date = '121021'
target_date = '121021'

tr_cells = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C09', 'C10']


data = du.Data(animal_id, date, datadir, tr_cells)
X = data.tr_data.to_numpy()
y = data.taste_event

# X = data.tr_data
# y = data.taste_event

def train_SVM():
    
    X = data.tr_data.to_numpy()
    y = data.taste_event
    
    scalar = preprocessing.StandardScaler().fit(X)
    scalar.transform(X)
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
    
    # Encode targets 
    X_le = preprocessing.LabelEncoder()
    y_le = preprocessing.LabelEncoder()
    
    y_train = X_le.fit_transform(y_train)
    y_test = y_le.fit_transform(y_test)
    
    clf = SVM.SupportVectorMachine(power=4, coef=1)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = func.accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Support Vector Machine", accuracy=accuracy)
    

#%%

if __name__ == "__main__":
    train_SVM()
    
    # data_test = du.Data(animal_id, target_date, datadir, tr_cells)
    # nn = m.NeuralNetwork(data, data_test)

    # nn.SVM()

