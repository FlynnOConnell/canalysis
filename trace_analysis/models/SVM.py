#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# models.py

Module (models): Classes/functions for NeuralNetwork module training.
"""

from __future__ import division
from typing import Optional
import logging
logger = logging.getLogger(__name__)

from sklearn import preprocessing
from sklearn.svm import (
    SVC,
    LinearSVC as LSVC)

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    ShuffleSplit,
    GridSearchCV,
    train_test_split
    )


from core import draw_plots
from models.scores import Scoring
from models import param_loop
from models.params import (
    svm_models_n_params,
    svm_models_n_params_small
    )

# %%

class SupportVectorMachine(object):
    
    """
    SVM Classifier. Default train/test size to 80% unless values given. 
    
    Args:
    -----------

        - _training_data (Traces): Instance of Data class for model training.
        - _eval_data (Traces): Instance of Data class for model evaluation.
        - **params (dict): Keyword args for train/test split and SVM. 
            
            Accepted **kwarg parameters: 
                
            TRAIN/TEST SPLIT
                -test_size (float): Between 0 and 1,  proportion of dataset to use for training. 
                    -default = 0.2
                -random_state (int): Any number to control exact split metric.
                    -defualt: 
                -stratify
            SVM
                -kernel (list | str)
                -c_range (list)
                -gamma_range (list)
                -mat (bool: confusion_matrix)
                -refit
                
    Returns:
    -----------
        None.

    """
    
    # Defaults 
    
    _cv =  StratifiedShuffleSplit(
           n_splits=1,
           train_size=0.6,
           random_state=0
           )
    
    _model = SVC()
    
    def __init__(self, _training_data, _eval_data, cv=_cv, model=_model, **params):
        
        
        self.scaler = preprocessing.StandardScaler()
        self.encoder = preprocessing.LabelEncoder()
        
        ## Training Data ---------------------------
        
        self.X = _training_data.tr_data  # x, [features]: taste responsive only
        self.X.reset_index(drop=True, inplace=True)
        
        self.y = _training_data.all_taste_trials.events  # y, [true]: array of true values
        
        ## Evaluation Data -------------------------
        
        self.X2 = _eval_data.tr_data  # x, [features]: taste responsive only
        self.X2.reset_index(drop=True, inplace=True)
        self.y2 = _eval_data.all_taste_trials.events  # y, [true]: array of true values

        # Encode/Decode y -------------------------
        self.encoder.fit(self.y)
        self.y = self.encoder.transform(self.y)
        self.y2 = self.encoder.transform(self.y2)
        
        # Training --------------------------------
        assert isinstance(cv,(StratifiedShuffleSplit, ShuffleSplit))
        self.cv = cv 
        
        self.model = model
        self.grid = None
        
        self.summary = {}
        self.train_scores = None
        self.eval_scores = None
        
        ## Helpers ---------------------------------

        self.classes = _training_data.tastant_colors_dict
    
        
    def _split(self, 
              X,
              y,
              train_size: float = 0.9,
              test_size: float = None,
              random_state=None,
              stratify=True,
              **cvparams):
                
        if stratify: 
            stratify=y
            
        logging.info(f'train_test_split completed with {self.cv}')
        
        train_index, test_index = next(self.cv.split(X, y))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        return X_train, X_test, y_train, y_test
            
        
    def _train_classifier(self, 
                         kernel=None, 
                         c_range = None, 
                         gamma_range = None, 
                         **params):
        """
        Return SVM classifier built from training data. Called from 
        "svm.fit" method and should not be called externally.
    
        Args:
            train_x (Iterable) : Features.
            train_y (Iterable) : Targets
            kernel (str): Function type for computing SVM().
                -'linear' (Default)
                -'rbf'
            c_range (list): Values of C parameter to run GridSearchCV.
            gamme_range (list):  Values of gamma parameter to run GridSearchCV.
            params (str): Instance of data for different session.

        Returns:
            None

        """
        from sklearn.pipeline import Pipeline      
        scalar = preprocessing.StandardScaler()
        svc = SVC()
        pipe = Pipeline([('scalar', scalar), ('svc', svc)])
        
        param_grid = {
            'svc__C': [0.1, 1, 10, 100, 150, 200, 500, 1000, 2000, 5000, 10000], 
            'svc__gamma':['scale', 'auto'],
            'svc__kernel': ['linear', 'rbf', 'poly']
            }  
        
        if 'c_range' in params:
            param_grid['svc__C'] = params['c_range']
        if 'gamma_range' in params:
            param_grid['svc__gamma'] = params['gamma_range']
        if 'kernel' in params:
            param_grid['svc__kernel'] = params['kernel']
                
        self.X_train, self.X_test, self.y_train, self.y_test = self._split(self.X, self.y)
        
        self.grid = GridSearchCV(pipe, param_grid=param_grid, cv=self.cv, verbose=False)
        self.grid.fit(self.X_train, self.y_train)
        
        # Store training results 
        self.summary['param_grid'] = param_grid
        self.summary['Best score - Train'] = self.grid.best_score_                
        self.summary['C'] = self.grid.best_params_['svc__C']
        self.summary['gamma'] = self.grid.best_params_['svc__gamma']
        self.summary['kernel'] = self.grid.best_params_['svc__kernel']


        print('**','-'*25,'*','-'*25,'**')
        print(f"Best params: {self.grid.best_params_}")
        print(f"Score: {self.grid.best_score_}")
        kernel = self.grid.best_params_['svc__kernel']
        c_best = self.grid.best_params_['svc__C']
        gamma_best = self.grid.best_params_['svc__gamma']
          
        self.model = SVC(C=c_best, kernel=kernel, gamma=gamma_best, verbose=False)
        
        return None


    def fit(self,
            params: Optional[str] = '',
            learning_curve:bool=False,
            **kwargs
            ) -> object:
            
        if 'mat' in kwargs: 
            mat = kwargs['mat']
        else:
            mat=True
        
        #### Get / Fit Model ####
        self._train_classifier()
        self.model.fit(self.X_train, self.y_train)
        
        y_pred = self.model.predict(self.X_test)
        
        self.train_scores = Scoring(y_pred, self.y_test, self.classes,
                                    descriptor='train', mat=mat, metrics=True)
        
        self.summary['train_scores'] = self.train_scores()                
        return None
    
    def validate(self, **kwargs):
        
        y_pred = self.model.predict(self.X2)
        
        if 'mat' in kwargs: 
            mat = kwargs['mat']
        else:
            mat=True
            
        self.eval_scores = Scoring(y_pred, self.y2, self.classes, descriptor='eval', mat=mat, metrics=True)
        
        return None
    
    @staticmethod
    def run_all_params(x, y, small: bool = False, normalize_x: bool = True):

        all_params = svm_models_n_params_small if small else svm_models_n_params

        return param_loop.big_loop(all_params,
                        preprocessing.StandardScaler().fit_transform(x) if normalize_x else x, y,
                        isClassification=True)
    
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
        if cv: 
            cv=cv
        else:
            cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
            
        estimator = estimator
        
        linSVC = LSVC()
        
        draw_plots.Plot.plot_learning_curve(
            linSVC,
            title, X, y, axes=axes[:, 0], ylim=(0, 1.01), cv=cv
        )
        
        kern = self.grid.best_params_['svc__kernel']
        gam = self.grid.best_params_['svc__gamma']
        C = self.grid.best_params_['svc__C']
        
        title = (f'SVC - kernel = {kern}, gamma = {gam}, C = {C}')
        if cv: 
            cv=cv
        else:
            # SVC is more expensive so decrease number of CV iterations:
            cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
            
        estimator = estimator
        draw_plots.Plot.plot_learning_curve(
            estimator, title, X, y, axes=axes[:, 1], ylim=(0, 1.01), cv=cv
        )
    
        plt.show()


if __name__ == "__main__":
    pass