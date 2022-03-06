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
        _eval (Traces): Instance of Data class for model evaluation.

    Returns:
    -----------
        None.

    """


    def __init__(self, _data, _eval):

        self._authenticate_input_data(_data, _eval)
        self.classes = _data.tastants

        ## Train/Validate Data ---------------------

        self.features = _data.tr_data  # x, [features]: taste responsive only 
        self.target = _data.all_taste_trials.tastant  # y, [true]: array of true values

        ## Evaluation Data -------------------------    

        self.features_eval = _eval.tr_data  # x, [features]: for evaluation 
        self.target_eval = _eval.all_taste_trials.tastant  # y, [true]: for evaluation

        ## Model and Scores ------------------------

        self.grid = None
        self.train_scores: dict = {}
        self.eval_scores: dict = {}

        ## Helpers ---------------------------------

        self.tastant_colors_dict = _data.tastant_colors_dict

    @staticmethod
    def _authenticate_input_data(_data, _eval) -> None:
        # verify inputs are members of main.Data class 
        if not isinstance(_data, du.Data):
            raise AttributeError('Input data must be an instance of the du.Data class')
        if not isinstance(_eval, du.Data):
            raise AttributeError('Input data must be an instance of the du.Data class')
        if not hasattr(_data, 'all_taste_trials'):
            raise AttributeError('Taste responsive cells not included in data class')
        if not hasattr(_data, 'all_taste_trials'):
            raise AttributeError('Taste responsive cells not included in data class')
        else:
            logging.info('NeuralNetwork instantiated.')


    def SVM(self,
            kernel: str = 'linear',
            params: Optional[str] = ''
            ) -> None:
        
        """
        
        Instance of NeuralNetwork for machine learning.
        Contains methods for training/testing, and models for ML.

        Current models:
            -SVM(Linear / rbf kernal)

        Args:
            kernel (str): Function type for computing SVM().
                -'linear' (Default)
                -'rbf'
            params (str): Instance of data for different session.

        Returns:
            Desired ML model.
        
        """

        from sklearn.svm import SVC
        
        # SVM doesnt take categorical data, so we encode into integers
        # 2 instances of LabelEncoder() class, so we can inverse_transform
        # each dataset later
        
        le_train = preprocessing.LabelEncoder()
        le_eval = preprocessing.LabelEncoder()

        target_encoded = le_train.fit_transform(self.target)
        target_eval_encoded = le_eval.fit_transform(self.target_eval)

        #### Build Model ####
        model = SVC(kernel=kernel)
        c_range = np.logspace(-2, 10, 13)
        if params:
            param_grid = params
        else:
            param_grid = dict(C=c_range)
        grid = model_selection.GridSearchCV(model, param_grid=param_grid)

        # TODO: Fix stratify parameter in train/test/split
        x_train, x_test, y_train, y_test = train_test_split(
            self.features,
            target_encoded,
            test_size=0.2,
            random_state=40,
            stratify=target_encoded)
        logging.info('train_test_split completed')

        # Fit training data only
        scalar = preprocessing.StandardScaler()
        scalar.fit(x_train)

        # Scale everything to the training data's fit
        scalar.transform(x_test)
        scalar.transform(self.features_eval)

        grid.fit(x_train, y_train)

        print(
            f"The best parameters are {grid.best_params_},"
            f" with a score of {grid.best_score_}"
        )

        model = grid.best_estimator_

        #### Fit model, get scores ###
        test_fit = model.predict(x_test)
        eval_fit = model.predict(self.features_eval)

        # Convert back to string labels
        self.target = le_train.inverse_transform(self.target)
        self.target_eval = le_eval.inverse_transform(self.target_eval)

        self.train_scores = Scoring(self.target,
                                    test_fit,
                                    self.classes,
                                    'train')
        self.eval_scores = Scoring(self.target_eval,
                                   eval_fit,
                                   self.classes,
                                   'eval')

        return None


@dataclass
class Scoring(object):

    def __init__(self,
                 pred,
                 true,
                 classes,
                 descriptor: Optional[str] = '',
                 mat: bool = True):
        logging.info('Scoring instance created.')

        # Input variables
        self.predicted = pred
        self.true = true
        self.classes = classes
        self.descriptor = descriptor

        # Report variables
        self.clf_report = self.get_report()
        self.accuracy = None
        self.recall = None
        self.precision = None
        self.f1 = None

        if mat:
            self.mat = self.get_confusion_matrix()

        if descriptor is None:
            logging.info('No descriptor')
            pass

    def get_report(self) -> pd.DataFrame:

        from sklearn.metrics import classification_report
        if self.descriptor:
            assert self.descriptor in ['train', 'test', 'eval']

        report = classification_report(
            self.true,
            self.predicted,
            target_names=self.classes,
            output_dict=True
        )
        report = pd.DataFrame(data=report).transpose()
        logging.info('Classification report created.')

        return report

    def get_metrics(self):

        from sklearn import metrics

        self.accuracy = metrics.accuracy_score(self.true, self.predicted)
        self.precision = metrics.precision_score(self.true, self.predicted, average='micro')
        self.recall = metrics.recall_score(self.true, self.predicted, average='micro')
        self.f1 = metrics.f1_score(self.true, self.predicted, average='micro')

    def get_confusion_matrix(self, caption: Optional[str] = '') -> object:

        mat = draw_plots.confusion_matrix(
            self.true,
            self.predicted,
            labels=self.classes,
            caption=caption)
        logging.info('Matrix created.')

        return mat


if __name__ == "__main__":
    pass
