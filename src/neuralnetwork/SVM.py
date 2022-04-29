#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# neuralnetwork.py

Module (neuralnetwork): Classes/functions for NeuralNetwork module training.
"""

from __future__ import division

import logging
from enum import Enum, auto
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    ShuffleSplit,
    GridSearchCV)
from sklearn.svm import SVC
from graphs.graph_utils.graph_funcs import plot_learning_curve
logger = logging.getLogger(__name__)


# %%

class Stage(Enum):
    TRAIN = auto()
    VAL = auto()


class DataHandler:

    def __init__(self, data, target):
        """
        Check and index specific data to feed into SVM. Accepted as input to sklearn.GridSearchCV().
        Features are the data used for regression and margin vectorizations.
        Labels (or targets, synonymous) are what the classifier is being trained on.

        Args:
            data (pd.DataFrame | np.ndarray): Features.
            target (pd.Series | np.ndarray): Labels/targets.

        Returns:
            None.
        """
        self.stage = 'train'
        self.data = np.array(data)
        self.target = np.array(target)

    def __getitem__(self, idx: int):
        """
        Indexer. Returns both feature and target values.
        
        Args:
            idx (int | Iterable[Any]): Indexors of any type.

        Returns:
            data[slice]: Indexed features.
            target[slice]: Indexed targets.
        """
        return self.data[idx], self.target[idx]


class SupportVectorMachine:

    def __init__(self, data, target):
        """
        Base class for svc.SVM neural network model.
        
        -Instances of OrdinalEncoder() is needed to "hide" true targets for learning, converting 
         a binary classification to [-1, 1] values or a multivariate classification to [-1, n-1]. 
        -Instances of StandardScaler() needed to scale data to a mean = 0 and stdev = 1. 
        
        These attributes will be shared for each session.
        Args:
            data (pd.DataFrame | np.ndarray): Indexors of any type.

        Returns:
            None.

        """
        self.summary: dict = {}
        self.TRAINDATA = DataHandler(data=data, target=target)
        self.EVALDATA = None
        self.labelencoder = preprocessing.OrdinalEncoder()
        self.scaler = preprocessing.StandardScaler()
        self.model = None
        self.grid = None

        self.trainset = {}

    @staticmethod
    def validate_shape(x, y):
        assert x.shape[0] == y.shape[0]

    @staticmethod
    def to_numpy(arg):
        if isinstance(arg, np.ndarray):
            return arg
        elif isinstance(arg, pd.Series):
            return arg.to_numpy()
        else:
            return np.array(arg)
        
        
    def split(self,
              train_size: float = 0.9,
              n_splits=1,
              **params
              ) -> None:
        """
        Split training dataset into train/test data.

        Args:
            train_size (float, optional): Proportion used for training. Defaults to 0.9.
            n_splits (int, optional): Number of iterations to split. Defaults to 1.
            **params (dict): Optional ShuffleSplit estimator params or custom labels/data.

        Returns:
            X_train (Iterable[Any]): Training features.
            x_test (Iterable[Any]): Testing features.
            Y_train (Iterable[Any]): Training labels.
            y_test (Iterable[Any]): Testing labels.
        """
        data = params.pop('data', self.TRAINDATA.data)
        target = params.pop('target', self.TRAINDATA.target)

        shuffle_split = StratifiedShuffleSplit(
            n_splits=n_splits,
            train_size=train_size,
            **params)
        
        train_index, test_index = next(shuffle_split.split(data, target))
        X_train, x_test = data[train_index], data[test_index]
        Y_train, y_test = target[train_index], target[test_index]

        self.trainset['X_train'] = X_train
        self.trainset['x_test'] = x_test
        self.trainset['Y_train'] = Y_train
        self.trainset['y_test'] = y_test
        self.trainset['encoded'] = 'no'
        return None

    def scale_encode(self, **kwargs):
        """
        Scale to mean = 0 and st.dev = 1 a train/split dataset.

        Args:
            *args (Iterable[Any]): Data provided from Split() method, or self provided from kwargs.
            **kwargs (dict): Custom train/test data. Must be supplied given names: 
                -X_train, x_test, Y_train, y_test
        Returns:
            None.
        """
        if not kwargs:
            X_train = self.trainset['X_train']
            x_test = self.trainset['x_test']
            Y_train = self.trainset['Y_train']
            y_test = self.trainset['y_test']
        else:
            X_train = kwargs['X_train']
            x_test = kwargs['x_test']
            Y_train = kwargs['Y_train']
            y_test = kwargs['y_test']
        # Get scaler for only training data, apply to training and test data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        x_test_scaled = self.scaler.transform(x_test)
        self.labelencoder.fit(Y_train.reshape(-1,1))
        Y_train_encoded = self.labelencoder.transform(Y_train.reshape(-1,1))
        y_test_encoded = self.labelencoder.transform(y_test.reshape(-1,1))
        self.trainset['X_train'] = self.to_numpy(X_train_scaled)
        self.trainset['x_test'] = self.to_numpy(x_test_scaled)
        self.trainset['Y_train'] = self.to_numpy(Y_train_encoded)
        self.trainset['y_test'] = self.to_numpy(y_test_encoded)
        self.trainset['encoded'] = 'yes'
        return None

    def optimize_clf(self,
                     X_train=None,
                     Y_train=None,
                     param_grid=None,
                     verbose=True,
                     refit=True,
                     **svcparams):
        if param_grid:
            param_grid = param_grid
        else:
            param_grid = {
                'C': [0.1, 1, 10, 100, 150, 200, 500, 1000],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf', 'poly']}
        if not X_train:
            X_train = self.trainset['X_train']
            Y_train = self.trainset['Y_train']
        else:
            X_train = X_train
            Y_train = Y_train

        svc = SVC(class_weight='balanced', **svcparams)
        self.grid = GridSearchCV(svc, param_grid=param_grid, refit=refit, verbose=verbose)
        self.grid.fit(X_train, Y_train.ravel())

        # Store training results 
        self.summary['param_grid'] = param_grid
        self.summary['Best score - Train'] = self.grid.best_score_
        self.summary['C'] = self.grid.best_params_['C']
        self.summary['gamma'] = self.grid.best_params_['gamma']
        self.summary['kernel'] = self.grid.best_params_['kernel']

        print('**', '-' * 20, '*', '-' * 20, '**')
        print(f"Best params: {self.grid.best_params_}")
        print(f"Score: {self.grid.best_score_}")
        kernel = self.grid.best_params_['kernel']
        c_best = self.grid.best_params_['C']
        gamma_best = self.grid.best_params_['gamma']

        self.model = SVC(C=c_best, kernel=kernel, gamma=gamma_best, verbose=False)
        return None


    def fit_clf(self,
                model=None,
                learning_curve: bool = False,
                **kwargs) -> object:
        """
        Fit classifier to training data.

        Args:
            **kwargs (dict): optional args for classifier.
        """
        X_train = self.trainset['X_train']
        Y_train = self.trainset['Y_train']
        x_test = self.trainset['x_test']
        if model: 
            self.model=model
        self.model.fit(X_train, Y_train)
        self.trainset['y_pred'] = self.model.predict(x_test)
        return None
    
    
    def predict_clf(self, x_test):
        x_test = self.trainset['x_test']
        self.trainset['y_pred'] = self.model.predict(x_test)
        return None


    def validate_clf(self):
        self.y_pred = self.model.predict(self.X2)
        return None
    

    def get_learning_curves(self,
                            estimator,
                            cv=None,
                            title: str = 'Learning Curve'):

        import matplotlib.pyplot as plt
        _, axes = plt.subplots(3, 2, figsize=(10, 15))

        X = self.X
        y = self.y

        # Cross validation with 50 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        title = title
        estimator = estimator
        plot_learning_curve(
            title, X, y, axes=axes[:, 0], ylim=(0, 1.01), cv=cv)
        kern = self.grid.best_params_['kernel']
        gam = self.grid.best_params_['gamma']
        C = self.grid.best_params_['C']
        title = (f'SVC - kernel = {kern}, gamma = {gam}, C = {C}')
        if cv:
            cv = cv
        else:
            # SVC is more expensive so decrease number of CV iterations:
            cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
        estimator = estimator
        plot_learning_curve(
            title, X, y, axes=axes[:, 1], ylim=(0, 1.01), cv=cv)

        plt.show()


if __name__ == "__main__":
    pass
