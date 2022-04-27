#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# scores.py

Module (models): Structures to score and keep scores for evaluated data. 

"""

from __future__ import division
from typing import Optional
from dataclasses import dataclass
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report)

from core import draw_plots
import pickle


def save(save_file_path, team):
    with open(save_file_path, 'wb') as f:
        pickle.dump(team, f)


def load(save_file_path):
    with open(save_file_path, 'rb') as f:
        return pickle.load(f)


@dataclass
class Scoring(object):

    def __init__(self,
                 pred,
                 true,
                 classes,
                 descriptor: Optional[str] = '',
                 mat: bool = False,
                 ):

        # Input variables
        self.report = None
        
        self.predicted = pred
        self.true = true
        self.classes = classes
        self.descriptor = descriptor
        
        # Report variables
        self.report = self.get_report()

        if mat:
            self.mat = self.get_confusion_matrix()
        
        if descriptor is None:
            logging.info('No descriptor')
            pass

    def get_report(self) -> pd.DataFrame:

        if self.descriptor:
            assert self.descriptor in ['train', 'test', 'eval']

        self.report = classification_report(
            self.true,
            self.predicted,
            target_names=self.classes,
            output_dict=True
        )
        report_df = pd.DataFrame(data=self.report).transpose()

        return report_df

    def get_confusion_matrix(self, caption: Optional[str] = '') -> object:

        mat = draw_plots.Plot.confusion_matrix(
            self.true,
            self.predicted,
            labels=self.classes,
            caption=caption)

        return mat

