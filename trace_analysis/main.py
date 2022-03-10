# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import pandas as pd
import logging

import data as du
from models.models import SupportVectorMachine

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
y = data.taste_events
x = data.all_taste_trials


def train_SVM():
    my_svm = SupportVectorMachine(data)


#     print ("Accuracy:", accuracy)
#     # print(y_pred)
#     # Reduce dimension to two using PCA and plot the results
#     Plot().plot_in_2d(X_test,
#                       y_pred,
#                       caption='Norm entire dataset pre-training',
#                       title="Support Vector Machine", accuracy=accuracy)
#     return y_test, y_pred

# y_test, y_pred = train_SVM()


# %%

if __name__ == "__main__":
    train_SVM()
