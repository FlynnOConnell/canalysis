#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# .../models/tracking.py

"""
from enum import Enum, auto
from typing import Protocol
from sklearn import preprocessing
import numpy as np
from sklearn import svm


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class ModelTracker(Protocol):

    def add_train_log(self, n_splits: int, train_size: float):
        """Train Descriptors"""

    def add_descriptor_log(self, descriptor: str):
        """Descriptor"""
        
    def add_range_params_log(self, range_params: dict):
        """Params"""

    def add_best_params_log(self, best_params: dict):
        """Model"""
        
    def add_best_model_log(self, model: svm.SVC):
        """Params"""

    def add_train_acc_log(self, train_acc: dict):
        """Params"""

    def add_test_acc_log(self, test_acc: dict):
        """Params"""
