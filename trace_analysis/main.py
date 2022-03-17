# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import pandas as pd
import logging

from models.SVM import SupportVectorMachine
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

# Fetch Data
train_data = CalciumData(animal_id, date, datadir, tr_cells)
val_data = CalciumData(animal_id, target_date, datadir, tr_cells)

# RUn SVM
svm = SupportVectorMachine(train_data, val_data)
svm.get_learning_curves()
svm.fit(mat=False)
svm.validate()
 
# Get SVM Results
results = svm.summary
eval_scores = svm.summary['eval_scores']
eval_report = svm.eval_scores.clf_report

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

X = svm.X
y_pre = svm.y
y = svm.encoder.inverse_transform(svm.y)
X['y'] = y_pre
X = X.astype(float)


X_out = X[(np.abs(stats.zscore(X)) < 3.5).all(axis=1)]

sns.scatterplot(
    x=X_out.iloc[:,0:4], y=X_out.iloc[:,5:9], hue=X_out['y'], marker="o", s=25, edgecolor="k", legend=True
).set_title("Taste Signal Values")
plt.show()

sns.scatterplot(
    x=X[:, 0], y=X[:, 1], hue=y, marker="o", s=25, edgecolor="k", legend=True
).set_title("Taste Signal Values")
plt.show()


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