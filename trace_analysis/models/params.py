#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#params.py

Module (models): Range of parameter values to loop through.
"""
import numpy as np
from sklearn.svm import SVC, LinearSVC, NuSVC


C = {'C': [1e-2, 0.1, 1, 5, 10, 100, 1000]}
C_small = {'C': [ 0.1, 1, 5]}
kernel = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
degree = {'degree': [1, 2, 3, 4, 5]}
gamma = {'gamma': list(np.logspace(-9, 3, 6)) + ['auto']}
coef0 = {'coef0': [0, 0.1, 0.3, 0.5, 0.7, 1]}
shrinking = {'shrinking': [True, False]}
max_iter = {'max_iter': [100, 300, 1000]}
max_iter_inf2 = {'max_iter': [100, 300, 500, 1000, -1]}
tol = {'tol': [1e-4, 1e-3, 1e-2]}
n_iter = {'n_iter': [5, 10, 20]}
nu = {'nu': [1e-4, 1e-2, 0.1, 0.3, 0.5, 0.75, 0.9]}
nu_small = {'nu': [1e-2, 0.1, 0.5, 0.9]}
penalty_12 = {'penalty': ['l1', 'l2']}

svm_models_n_params = [
    (SVC,
     {**C, **kernel, **degree, **gamma, **coef0, **shrinking, **tol, **max_iter_inf2}),

    (NuSVC,
     {**nu, **kernel, **degree, **gamma, **coef0, **shrinking, **tol
      }),

    (LinearSVC,
     { **C, **penalty_12, **tol, **max_iter,
       'loss': ['hinge', 'squared_hinge'],
       })
]

svm_models_n_params_small = [
    (SVC,
     {**kernel, **degree, **shrinking
      }),

    (NuSVC,
     {**nu_small, **kernel, **degree, **shrinking
      }),

    (LinearSVC,
     { **C_small,
       'penalty': ['l2'],
       'loss': ['hinge', 'squared_hinge'],
       })
]
