#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# models.py
"""
#models.py

Module: Classes/functions for NeuralNetwork module training.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from typing import Optional, Generic

from pandas import DataFrame
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.model_selection import train_test_split


import data as du
from utils import funcs, draw_plots

logger = logging.getLogger(__name__)


# %%

def model_to_csv(df, name) -> None:
    resultsdir = '/Users/flynnoconnell/Desktop'
    with pd.ExcelWriter(resultsdir
                        + '/'
                        + name
                        + '.xlsx') as writer:
        df.to_excel(
            writer, sheet_name=name, index=False)
    return None


# Issue here with design; Immutable vs mutable objects. Do we want an immutable, untouchable dataset
# that we copy and manupulate to make new objects. Or do we want to change the object as we go and create 
# a new instance with the original dataset if need be. With SVM being on the upper limit ofhow many models were going 
# to implement, probably doesn't matter. Just needs to be consistent. 


def get_encoder(y):
    """
    SVM() is not meant for categorical data.
    We need to convert target sets from categories to integers:
        Artsal : 1
        MSG: 2
        Quinine: 3 etc. 
        Args:
            y (np.ndarray): 1d array with categorical data to encode
        Returns:
             Generic[le]: Instance of preprocessing.LabelEncoder()
             -This instance is used to inverse_transform() later on.
    """
    le = preprocessing.LabelEncoder()
    le.fit_transform(y)

    return le, y


def pca(features,
        color_dict,
        scatter_colors,
        graph: Optional[bool] = True):
    scalar = preprocessing.StandardScaler()

    scalar.fit_transform(features)
    pca = PCA(n_components=0.95)
    df = pca.fit_transform(features)
    variance = np.round(
        pca.explained_variance_ratio_ * 100, decimals=1)
    labels = [
        'PC' + str(x) for x in range(1, len(variance) + 1)]

    df = pd.DataFrame(df, columns=labels)

    if graph:
        draw_plots.scatter(
            df,
            color_dict,
            df_colors=scatter_colors,
            title='Taste Responsive Cells',
            caption='Datapoint include only taste-trials. Same data as used in SVM')
    else:
        return df, variance, labels


class NeuralNetwork(object):

    def __init__(self, _data, _eval):
        """
        Instance of NeuralNetwork for machine learning.
        Contains methods for training/testing, and models for ML.
        Arguments are modified in-class. Create a new instance for new models.

        Current models:
            -SVM(Linear / rbf kernal)

        Args:
            _data (Traces): Instance of Data class for model training.
            _eval (Traces): Instance of Data class for model evaluation.

        Returns:
            None.

        """

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
        self.grid = model_selection.GridSearchCV(model, param_grid=param_grid)

        # TODO: Fix stratify parameter in train/test/split
        # Setting stratify=True breaks the split due to "Singleton array(True)
        # is not a valid collection?
        x_train, x_test, y_train, y_test = train_test_split(
            self.features,
            target_encoded,
            test_size=0.2,
            random_state=40)
        logging.info('train_test_split completed')

        # Fit training data only
        scalar = preprocessing.StandardScaler()
        scalar.fit(x_train)

        # Scale everything to the training data's fit
        scalar.transform(x_test)
        scalar.transform(self.features_eval)

        self.grid.fit(x_train, y_train)

        print(
            f"The best parameters are {grid.best_params_},"
            f" with a score of {grid.best_score_}"
        )

        model = self.grid.best_estimator_

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

    def get_report(self) -> DataFrame:

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
