#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# param_loop.py

Module (models): Main loop functions for parameter hypertuning.
"""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit as sss, ShuffleSplit as ss, GridSearchCV
from time import time
from tabulate import tabulate
import logging

# %% Initialize data

logger = logging.getLogger(__name__)
logger.info(f'{__name__} called.')


def timeit(klass, params, x, y):
    """
    time in seconds
    """

    start = time()
    clf = klass(**params)
    clf.fit(x, y)

    return time() - start


def cv_clf(x, y,
           test_size=0.2, n_splits=5):
    """
    an iterator of cross-validation groups with upsampling
    :param x:
    :param y:
    :param test_size:
    :param n_splits:
    :return:
    """

    sss_obj = sss(test_size)
    sss_obj.split(x, y)

    return sss_obj


def cv_reg(x, test_size=0.2, n_splits=5):
    return ss(n_splits, test_size).split(x)


def big_loop(models_n_params, x, y, isClassification,
             test_size=0.2, n_splits=5, random_state=None, doesUpsample=True,
             scoring=None,
             verbose=False):
    """
    runs through all model classes with their perspective hyper parameters
    :param models_n_params: [(model class, hyper parameters),...]
    :param isClassification: whether it's a classification or regression problem
    :type isClassification: bool
    :param scoring: by default 'accuracy' for classification; 'neg_mean_squared_error' for regression
    :return: the best estimator, list of [(estimator, cv score),...]
    
    """

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        def cv_():
            return cv_clf(x, y, test_size, n_splits) \
                if isClassification \
                else cv_reg(x, test_size, n_splits)

        res = []
        scoring = 'accuracy'
        print('Scoring criteria:', scoring)

        for i, (clf_Klass, parameters) in enumerate(models_n_params):
            print('-' * 15, 'model %d/%d' % (i + 1, len(models_n_params)), '-' * 15)
            print(clf_Klass.__name__)

            clf_search = GridSearchCV(clf_Klass(), parameters)

            clf_search.fit(x, y)

            timespent = timeit(clf_Klass, clf_search.best_params_, x, y)

            print('best score:', clf_search.best_score_, 'time/clf: %0.3f seconds' % timespent)

            print('best params:')
            print(clf_search.best_params_)

            if verbose:
                print('validation scores:', clf_search.cv_results_['mean_test_score'])
                print('training scores:', clf_search.cv_results_['mean_train_score'])

            res.append((clf_search.best_estimator_, clf_search.best_score_, timespent))

        print('=' * 60)
        print(tabulate([[m.__class__.__name__,
                         '%.3f' % s, '%.3f' % t] for m, s, t in res], headers=['Model', scoring, 'Time/clf (s)']))

        winner_ind = np.argmax([v[1] for v in res])
        winner = res[winner_ind][0]
        print('=' * 60)
        print('The winner is: %s with score %0.3f.' % (winner.__class__.__name__, res[winner_ind][1]))

        return winner, res
