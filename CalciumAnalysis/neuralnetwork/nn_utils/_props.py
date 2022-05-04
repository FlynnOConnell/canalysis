#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# _props.py

Module(nn_utils): Boilerplate property getter/setters for SVM model.
    
"""
from neuralnetwork.nn_utils._validate import _validate
from neuralnetwork.nn_utils.datahandler import DataHandler


class _props:
    @property
    def evaldata(self):
        return self._evaldata
    
    @evaldata.setter
    def evaldata(self, handlerargs):
        try:
            data, target, stage = handlerargs
        except: 
            raise AttributeError('Must pass an iterable with 3 values')
        self._evaldata = DataHandler(data=data, target=target, stage=stage)
        print(f'Eval data set: {handlerargs}')
       
    @property
    def cv(self):
        return self._cv
    
    @cv.setter
    def cv(self, new_cv):
        self._cv = new_cv
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, new_model):
        if new_model:
            _validate._check_fitted(new_model)
            print(f'Model set: {new_model}')
        self._model = new_model
