#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# scores.py

Module (neuralnetwork): Structures to score and keep scores for evaluated data.

"""

from __future__ import division

import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from sklearn.metrics import classification_report
from graphs.plots import Plot
import pickle
logger = logging.getLogger(__name__)


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
                 mat: bool = False):
        """
        Class to manage scoring variables from fitted classifiers. 

        Parameters
        ----------
        pred : ndarray
            Fitted model's "predicted" output.
        true : ndarray
            Descriptors of each predictable value.
        classes : Iterable
            Descriptors of each predictable value.
        descriptor : Optional[str], optional
            Description of what input was used. 
        mat : bool, optional
            Whether to output a Confusion Matrix. The default is False.

        Returns
        -------
        None.
        """
        # Input variables
        self.report = self.get_report()
        self.predicted = pred
        self.true = true
        self.classes = classes
        self.descriptor = descriptor
        if mat:
            self.mat = self.get_confusion_matrix()
        if descriptor is None:
            logging.info('No descriptor')
            pass

    def get_report(self) -> pd.DataFrame:
        """ Get classification report"""
        if self.descriptor:
            assert self.descriptor in ['train', 'test', 'eval']
        self.report = classification_report(
            self.true,
            self.predicted,
            target_names=self.classes,
            output_dict=True)
        report_df = pd.DataFrame(data=self.report).transpose()
        return report_df

    def get_confusion_matrix(self, caption: Optional[str] = '') -> object:
        """ Get confusion matrix"""
        mat = Plot.confusion_matrix(
            self.true,
            self.predicted,
            labels=self.classes,
            caption=caption)
        return mat
