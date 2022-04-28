#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# models.py

Module (models): Classes/functions for NeuralNetwork module training.
"""

from __future__ import division

from typing import Optional, Iterable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd
import logging

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.svm import (
    SVC,
    LinearSVC)

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    ShuffleSplit,
    GridSearchCV,
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report
    )

from graphs.plot import Plot
from models.utils.tracking import Stage
logger = logging.getLogger(__name__)


# %%


    
class DataHandler(object):
    
    def __init__(self, stage, data, target = None):
        """
        Check and index specific data to feed into SVM. Accepted as input to sklearn.GridSearchCV().
        Features are the data used for regression and margin vectorizations.
        Labels (or targets, synonymous) are what the classifier is being trained on.

        Args:
            data (pd.DataFrame | np.ndarray): Features.
            target (pd.Series | np.array): Labels/targets.

        Returns:
            None.
        """
        self.stage = Stage.VAL if target is None else Stage.TRAIN
        self.data = data
        self.target = target
        if (self.data.shape[0]) != (self.target.shape[0]):
            raise Exception('Wrong')


    def __getitem__(self, idx: int):
        """
        Indexer. Returns both feature and target values.
        
        Args:
            x (int | Iterable[Any]): Indexors of any type.

        Returns:
            data[slice]: Indexed features.
            target[slice]: Indexed targets.
        """
        return self.data[idx], self.target[idx]


class modelSVM(object):
    
    def __init__(self, data, target):
        """
        Base class for SVM. 
        
        -Instances of OrdinalEncoder() is needed to "hide" true targets for learning, converting 
         a binary classification to [-1, 1] values or a multivariate classification to [-1, n-1]. 
        -Instances of StandardScaler() needed to scale data to a mean = 0 and stdev = 1. 
        
        These attributes will be shared for each session.

        Returns:
            None.

        """
        self.TRAINDATA = DataHandler('train', data, target)
        self.TESTDATA = None
        self.encoder = preprocessing.OrdinalEncoder()
        self.scaler = preprocessing.StandardScaler()
        self.model = None
        self.grid = None
        
        self.trainset = {}
            
        
    def validate_shape(self, x, y):
        assert x.shape[0] == y.shape[0]
        
        
    def split(self,
              train_size: float = 0.9,
              test_size: float = None,
              n_splits=1,
              stratify=True,
              **params
              ):
        """
        Split training dataset into train/test data.

        Args:
            train_size (float, optional): Proportion used for training. Defaults to 0.9.
            test_size (float, optional): Proportion used for testing. Defaults to None.
            n_splits (int, optional): Number of iterations to split. Defaults to 1.
            stratify (bool, optional): Split method. Defaults to True.
            **params (dict): Optional ShuffleSplit estimator params or custom labels/data. 

        Returns:
            X_train (Iterable[Any]): Training features.
            X_test (Iterable[Any]): Testing features. 
            y_train (Iterable[Any]): Training labels.
            y_test (Iterable[Any]): Testing labels.

        """      
        data = params.pop('data', self.TRAINDATA.data)
        y = params.pop('y', self.TRAINDATA.target)
        if stratify:
            stratify = y
        else:
            stratify=False
        shuffle_split = StratifiedShuffleSplit(
            n_splits=n_splits,
            train_size=train_size,
            **params)
        train_index, test_index = next(shuffle_split.split(data, y))
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        self.trainset['X_train'] = X_train
        self.trainset['X_test'] = X_test
        self.trainset['y_train'] = y_train
        self.trainset['y_test'] = y_test
        return self.scale(X_train, X_test, y_train, y_test)
    #models
    def scale_encode(self, *args, **kwargs):
        """
        Scale the split data. 

        Args:
            *args (Iterable[Any]): Data provided from Split() method, or self provided from kwargs.
            **kwargs (dict): Custom train/test data. Must be supplied given names: 
                -X_train, X_test, y_train, y_test
        Returns:
            None.

        """
        X = kwargs.pop('X_train',args[0])
        x = kwargs.pop('X_test', args[1])
        Y = kwargs.pop('y_train', args[2])
        y = kwargs.pop('y_test', args[3])
        
        super().scaler.fit(X)
        X = super().scaler.transform(X)
        x = super().scaler.transform(x)
        super().encoder.fit(Y)
        Y = super().encoder.transform(Y)
        y = super().encoder.transform(y)


    def optimize(self,
                multi_class='auto',
                max_iter=None,
                solver='lbfgs',
                class_weight='balanced',
                **kwargs):
        
        scv__C = kwargs.pop('scv__C',  [0.1, 1, 10, 100, 150, 200, 500, 1000])
        svc__gamma = kwargs.pop('svc__gamma', ['scale', 'auto'])
        svc__kernel = kwargs.pop('svc__kernel', ['linear', 'rbf', 'poly'])
        svc__epsilon = kwargs.pop('svc__epsilon', [0.1,0.2,0.5,0.3])
        verbose = kwargs.pop('verbose', True)
        
        param_grid = {
            'svc__C': scv__C,
            'svc__gamma': svc__gamma,
            'svc__kernel': svc__kernel,
            'svc__epsilon': svc__epsilon
            }
        
        param_grid = kwargs.pop('param_grid', param_grid)
        
        svc = SVC(max_iter=max_iter, class_weight='balanced')
        self.grid = GridSearchCV(svc, param_grid=param_grid, verbose=verbose)
        self.grid.fit(self.trainset['X_train'], self.trainset['y_train'])

        # Store training results 
        self.summary['param_grid'] = param_grid
        self.summary['Best score - Train'] = self.grid.best_score_
        # self.summary['kernel'] = self.grid.best_params_['kernel']
        self.summary['C'] = self.grid.best_params_['svc__C']
        self.summary['gamma'] = self.grid.best_params_['svc__gamma']
        self.summary['kernel'] = self.grid.best_params_['svc__kernel']

        print('**', '-' * 20, '*', '-' * 20, '**')
        print(f"Best params: {self.grid.best_params_}")
        print(f"Score: {self.grid.best_score_}")
        kernel = self.grid.best_params_['svc__kernel']
        c_best = self.grid.best_params_['svc__C']
        gamma_best = self.grid.best_params_['svc__gamma']

        best_clf = SVC(C=c_best, kernel=kernel, gamma=gamma_best, verbose=False)

        return best_clf

    def fit(self,
            params: Optional[str] = '',
            learning_curve: bool = False,
            **kwargs
            ) -> object:

        if 'mat' in kwargs:
            mat = kwargs['mat']
        else:
            mat = True

        x_train = self.X_train
        y_train = self.y_train

        #### Get / Fit Model ####

        self.model = self._get_classifier(self, x_train, y_train)
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)

        self.train_scores = Scoring(y_pred, self.y_test, self.classes,
                                    descriptor='train', mat=mat, metrics=True)

        self.summary['train_scores'] = self.train_scores
        self.summary['train_acc'] = self.train_scores.accuracy

        return None

    def validate(self, **kwargs):

        y_pred = self.model.predict(self.X2)
        if 'mat' in kwargs:
            mat = kwargs['mat']
        else:
            mat = True

        self.eval_scores = Scoring(y_pred, self.y2, self.classes, descriptor='eval', mat=mat, metrics=True)

        self.summary['eval_scores'] = self.eval_scores
        self.summary['Best score - test'] = self.eval_scores.accuracy

        return None


    def get_learning_curves(self,
                            estimator,
                            cv=None,
                            title: str = ''):

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(10, 15))

        X = self.X
        y = self.y

        # Cross validation with 50 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        title = "Learning Curves (LinearSVC"
        estimator = estimator

        
        Plot.plot_learning_curve(
            title, X, y, axes=axes[:, 0], ylim=(0, 1.01), cv=cv
        )

        kern = self.grid.best_params_['svc__kernel']
        gam = self.grid.best_params_['svc__gamma']
        C = self.grid.best_params_['svc__C']

        title = (f'SVC - kernel = {kern}, gamma = {gam}, C = {C}')
        if cv:
            cv = cv
        else:
            # SVC is more expensive so decrease number of CV iterations:
            cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

        estimator = estimator
        plot_learning_curve(
            title, X, y, axes=axes[:, 1], ylim=(0, 1.01), cv=cv
        )

        plt.show()


if __name__ == "__main__":
    pass