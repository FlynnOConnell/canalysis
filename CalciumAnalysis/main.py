# -*- coding: utf-8 -*-
"""
#main_nn.py

Module: Main code execution. 
        Note: Neural network requires separate main.py in neuralnetwork subpackage.
"""

from __future__ import annotations

import os

import pandas as pd
import yaml

import utils.funcs
from utils.wrappers import log_time
from analysis.principal_components import get_pca
from analysis.process_data import ProcessData
from calcium_data import CalciumData
from data_utils.file_handler import FileHandler
from graphs.plot import pca_scatter


save_dir = r'C:\Users\flynn\Desktop\figs'
# save_dir = "C:/Users/dilorenzo/Desktop/CalciumPlots/"

__file__ = "main.py"
__location__ = os.path.realpath(
        os.path.join(
                os.getcwd(),
                os.path.dirname(__file__)))


# %%

def get_params():
    with open(os.path.join(__location__, 'params.yaml'),
              'rb') as f:
        parameters = yaml.safe_load(f.read())
    return Params(parameters)


class Params:
    """@DynamicAttrs"""

    def __init__(self, parameter_dict):
        for key in parameter_dict:
            setattr(self, key, parameter_dict[key])


def reorder(tracedata):
    dflist = ['C24', 'C23', 'C20', 'C05', 'C06', 'C07', 'C00', 'C26', 'C22', 'C09',
              'C10', 'C17', 'C11', 'C21', 'C04', 'C01', 'C02', 'C03', 'C14', 'C15',
              'C16', 'C27', 'C25', 'C18', 'C19', 'C12', 'C13', 'C08']
    tracedata.reorder(dflist)


def pca(df, color, param):
    mypca = get_pca(df, numcomp=3).pca_df
    scatter = pca_scatter(mypca, color, color_dict=param.Colors, edgecolors=None, s=20)
    return None


@utils.wrappers.log_time
def initialize_data(_filehandler: FileHandler, adjust: int = None):
    return CalciumData(_filehandler, color_dict=params.Colors, adjust=adjust)


def statistics(_data, _dir) -> pd.DataFrame | None:
    return ProcessData(_data)


def heatmap_loops(_data, anal, cols):
    for hm in _data.eatingdata.loop_eating(cols=cols, save_dir=save_dir):
        my_hm = hm


if __name__ == "__main__":
    params = get_params()

    filehandler = FileHandler(
            params.Session['animal'],
            params.Session['date'],
            params.Directory['data'],
            params.Filenames['traces'],
            params.Filenames['events'],
            params.Filenames['gpio'],
            params.Filenames['eating'])
    data = initialize_data(filehandler, adjust=34)
    save_dir = r'C:\Users\flynn\Desktop\figs'
    reorder(data.tracedata)
    for f in data.eatingdata.generate_eating_heatmap(save_dir='', title=''):
        f.show()


