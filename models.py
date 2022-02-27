#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# models.py
"""
Created on Tue Feb 22 15:39:48 2022

@author: flynnoconnell
"""
import logging
import data_utils as du
import optunity
import optunity.metrics
import Func as func
import Plots as plot
from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#%%

def model_to_csv(df, name) -> None:
    resultsdir = '/Users/flynnoconnell/Desktop'
    with pd.ExcelWriter(resultsdir
                        + '/'
                        + name
                        +'.xlsx') as writer:
        df.to_excel(
            writer, sheet_name=name, index=False)
    return None

class NeuralNetwork(object):

    def __init__(self, _data, _eval):

        self._authenticate_input_data(_data, _eval)
        
        self.feature_names = list(_data.tr_cells)
        self.classes = _data.tr_colors
        
        
        ## Model Information ----------------

        self.model_info = {
            'Type': '',
            'Pre-process': '',
            'Estimator':''
            }

        ## Train/Validate Data ---------------------
        
        self.features = _data.tr_data
        
        self.labels = _data.tastants
        self.target = _data.taste_trials.tastant
        
        ## Evaluation Data -------------------------
        
        self.feature_names_eval = list(_eval.tr_cells)
        self.features_eval = _eval.tr_data
        
        self.labels_eval= _eval.tastants
        self.target_eval = _eval.taste_trials.tastant
        
        self.clf = None
        
        ## Scores ----------------------------
                
        self.train_params_ = None
        self.train_report = None
        self.train_mat = None
        self.train_df = None

        self.train_scores: dict = {}
        self.eval_scores: dict = {}
        
        
        self.eval_grid = None
        self.eval_report = None
        self.eval_mat = None
        self.eval_df = None
        
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.F1 = None
        
        ## Helpers ---------------------------
        
        self.taste_colors_dict = _data.colors_dict



    @staticmethod
    def _authenticate_input_data(_data, _eval) -> None:
        # verify inputs are members of main.Data class 
        if not isinstance(_data, du.Data):
            raise AttributeError('Input data must be an instance of the du.Data class')
        if not isinstance(_eval, du.Data):
            raise AttributeError('Input data must be an instance of the du.Data class')
        else:
            logging.info('NeuralNetwork instantiated.')
                
    def pca(self, graph: bool = True):
        
        ## TODO: Make this work with test/eval datasets both 
        import pandas as pd
        
        data = ss().fit_transform(self.features)
        pca = PCA(n_components=self.num_features)
        df = pca.fit_transform(data)
        variance = np.round(
            pca.explained_variance_ratio_ * 100, decimals=1)
        labels = [
            'PC' + str(x) for x in range(1, len(variance) + 1)]
        
        to_plot = pd.DataFrame(data, columns=labels)
        
        if graph: 
            plot.scatter(
                to_plot,
                self.taste_colors_dict,
                df_colors = self.classes,
                title= 'Taste Responsive Cells', 
                caption= 'Datapoint include only taste-trials. Same data as used in SVM' )
        else:
            return df, variance, labels
        
    def train_test(self, 
                   test_size: float = 0.3,
                   random_state: int = 109
                   ):
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(
            self.features,
            self.target,
            test_size=test_size,
            random_state=random_state)
        logging.info('train_test_split completed')
        
        return x_train, x_test, y_train, y_test

    
    def get_scores(self):
        
        example = Scoring(self.target_eval,
                               self.eval_fit,
                               self.labels_eval,
                               'eval')
        return example

    
    def SVM(self,
            kernel: str = 'linear',
            params: str = ''
            ) -> None:
        
        
        """
        Support vector machine classifier for calcium data. 
        
        """

        assert len(self.feature_names) == len(self.feature_names_eval)

        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import StandardScaler        
        from sklearn.svm import SVC
        from sklearn import metrics
        from sklearn.pipeline import Pipeline
        
        self.model_info['kernal'] = 'linear'
        self.model_info['Type'] = 'SVM'
        self.model_info['pre-process'] = 'svm, svc'
                
        #### Build Model ####
        #####################
        
        ss = StandardScaler()
        model = make_pipeline(ss, SVC(kernel=kernel))
        
        if params is not None: 
            param_grid = params
        else:
            param_grid = {'svc__C': [1, 5, 10, 50, 100, 200]}
                      # 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
        
        self.clf = GridSearchCV(model, param_grid)
        
        #### Train/Fit Model ####
        #########################

        x_train, x_test, y_train, y_test = self.train_test()

        self.clf.fit(x_train, y_train)
        
        model = self.clf.best_estimator_
        
        y_fit = model.predict(x_test) 
        
        self.train_scores = Scoring(self.target, y_fit, self.labels)

        #### Get Scores ####
        ####################
        
        self.train_report = classification_report(
            y_test,
            y_fit,
            target_names=self.labels,
            output_dict=True
            )
        
        self.train_mat = plot.confusion_matrix(
            y_test,
            y_fit,
            labels=self.labels)
        
        self.train_df = pd.DataFrame(data=self.train_report).transpose()
        
        # model_to_csv(self.train_df, 'train')

        #### Evaluate Model ####
        ########################

        eval_fit = model.predict(self.features_eval)
        
        
        self.eval_scores = Scoring(self.target_eval, eval_fit, self.labels_eval)
        
        
        ## Scores
        
    
        self.accuracy = metrics.accuracy_score(self.target_eval, 
                                               self.eval_fit)
        
        self.precision =  metrics.precision_score(self.target_eval,
                                                  self.eval_fit,
                                                  average='micro')
        
        self.recall =  metrics.recall_score(self.target_eval,
                                            self.eval_fit,
                                            average='micro')
        self.F1 =  metrics.f1_score(self.target_eval,
                                    self.eval_fit,
                                    average='micro')

        # model_to_csv(self.train_df, 'test')

        return None

@dataclass
class Scoring(object):
    
    def __init__(self, 
                 pred,
                 true,
                 labels,
                 descriptor,
                 mat: bool=True):
        logging.info('Scoring instance created.')
        self.predicted = pred
        self.true = true
        self.labels = labels
        self.descriptor = descriptor
        self.report = self.get_report()
        if mat:
            self.mat = self.get_confusion_matrix()
            
        if descriptor is None: 
            pass
    
        
    def get_report(self) -> dict:
        
        from sklearn.metrics import classification_report
        if self.descriptor:
            assert self.descriptor in ['train' , 'test', 'eval']
        
        report = classification_report(
                self.true,
                self.predicted,
                target_names=self.labels,
                output_dict=True
                )
        report = pd.DataFrame(data=report).transpose()

        logging.info('Report created')
        
        return report

    def get_confusion_matrix(self,
                             caption: str='') -> object:
        
        mat = plot.confusion_matrix(
            self.true,
            self.predicted,
            labels=self.labels)
        logging.info('Matrix created')

        
        return mat
        
        
                
if __name__ == "__main__":

    pass
 

        




    
