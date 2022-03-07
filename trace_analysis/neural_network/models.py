#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# models.py
"""
#models.py

Module: Classes/functions for NeuralNetwork module training.
"""

from __future__ import division
from typing import Optional, Generic
from dataclasses import dataclass

import pandas as pd
import numpy as np
import logging

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.model_selection import train_test_split

import data as du
from utils import funcs

logger = logging.getLogger(__name__)


# %%

class SupportVectorMachine(object):
    """
    SVM Classifier.
    Uses sklearn.svm SVC() 

    Args:
    -----------       
    
        _data (Traces): Instance of Data class for model training.

    Returns:
    -----------
        None.

    """


    def __init__(self, _data):
        
        self.scaler = preprocessing.StandardScaler()
        self.encoder = preprocessing.LabelEncoder()
        
        ## Train/Validate Data ---------------------

        self.X = _data.tr_data  # x, [features]: taste responsive only 
        self.y = _data.all_taste_trials.events  # y, [true]: array of true values

        ## Helpers ---------------------------------

        self.tastant_colors_dict = _data.tastant_colors_dict


    def train(self, test_size: float = 0.2, random_state: int = 80): 
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y)
        logging.info('train_test_split completed')

        
        # Scale everything to the training data's fit
        self.scaler.fit(X_train)

        self.scaler.transform(X_test)
        self.scaler.transform()
        

#     def prep_data(self): 
        
#         encoder = preprocessing.LabelEncoder()
#         y = encoder.fit_transform(self.y)
#         assert y in [0,1,2,3,4,5]
        
#         return y
    
#     def SVM(self,
#             kernel: str = 'linear',
#             params: Optional[str] = ''
#             ) -> None:
        
#         """
        
#         Instance of NeuralNetwork for machine learning.
#         Contains methods for training/testing, and models for ML.

#         Current models:
#             -SVM(Linear / rbf kernal)

#         Args:
#             kernel (str): Function type for computing SVM().
#                 -'linear' (Default)
#                 -'rbf'
#             params (str): Instance of data for different session.

#         Returns:
#             Desired ML model.
        
#         """

#         from sklearn.svm import SVC

#         #### Build Model ####
#         model = SVC(kernel=kernel)
#         c_range = np.logspace(-2, 10, 13)
#         if params:
#             param_grid = params
#         else:
#             param_grid = dict(C=c_range)
#         grid = model_selection.GridSearchCV(model, param_grid=param_grid)


#         # Fit training data only

#         # grid.fit(x_train, y_train)

#         print(
#             f"The best parameters are {grid.best_params_},"
#             f" with a score of {grid.best_score_}"
#         )

#         model = grid.best_estimator_

#         #### Fit model, get scores ###
#         test_fit = model.predict(x_test)
#         eval_fit = model.predict(self.features_eval)

#         # Convert back to string labels
#         self.target = le_train.inverse_transform(self.target)
#         self.target_eval = le_eval.inverse_transform(self.target_eval)

#         self.train_scores = Scoring(self.target,
#                                     test_fit,
#                                     self.classes,
#                                     'train')
#         self.eval_scores = Scoring(self.target_eval,
#                                    eval_fit,
#                                    self.classes,
#                                    'eval')

#         return None


# @dataclass
# class Scoring(object):

#     def __init__(self,
#                  pred,
#                  true,
#                  classes,
#                  descriptor: Optional[str] = '',
#                  mat: bool = True):
#         logging.info('Scoring instance created.')

#         # Input variables
#         self.predicted = pred
#         self.true = true
#         self.classes = classes
#         self.descriptor = descriptor

#         # Report variables
#         self.clf_report = self.get_report()
#         self.accuracy = None
#         self.recall = None
#         self.precision = None
#         self.f1 = None

#         if mat:
#             self.mat = self.get_confusion_matrix()

#         if descriptor is None:
#             logging.info('No descriptor')
#             pass

#     def get_report(self) -> pd.DataFrame:

#         from sklearn.metrics import classification_report
#         if self.descriptor:
#             assert self.descriptor in ['train', 'test', 'eval']

#         report = classification_report(
#             self.true,
#             self.predicted,
#             target_names=self.classes,
#             output_dict=True
#         )
#         report = pd.DataFrame(data=report).transpose()
#         logging.info('Classification report created.')

#         return report

#     def get_metrics(self):

#         from sklearn import metrics

#         self.accuracy = metrics.accuracy_score(self.true, self.predicted)
#         self.precision = metrics.precision_score(self.true, self.predicted, average='micro')
#         self.recall = metrics.recall_score(self.true, self.predicted, average='micro')
#         self.f1 = metrics.f1_score(self.true, self.predicted, average='micro')

#     def get_confusion_matrix(self, caption: Optional[str] = '') -> object:

#         mat = draw_plots.confusion_matrix(
#             self.true,
#             self.predicted,
#             labels=self.classes,
#             caption=caption)
#         logging.info('Matrix created.')

#         return mat


if __name__ == "__main__":
    pass
