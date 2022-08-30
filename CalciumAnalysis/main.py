# -*- coding: utf-8 -*-

"""
#main_nn.py

Module: Main code execution. 
        Note: Neural network requires separate main.py in neuralnetwork subpackage.
"""
from __future__ import annotations

import logging

import graphs.plot
from analysis.analysis_utils import ca_pca
from calcium_data import CalciumData
from data_utils.file_handler import FileHandler
from taste_data import TasteData
from analysis.process_data import ProcessData
import pandas as pd
import faulthandler
import time
import numpy as np
import utils.funcs
from utils.wrappers import log_time
from graphs import plot
faulthandler.enable()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
save_dir = r'C:\Users\flynn\Desktop\figs'
# save_dir = "C:/Users/dilorenzo/Desktop/CalciumPlots/"
# %% Some examples on usage of different modules.

color_dict = {
    "Peanut": "darkorange",
    "Chocolate": "saddlebrown",
    "NaCl": "green",
    "Quinine": "red",
    "Acid": "yellow",
    "Sucrose": "purple",
    "Eating": "blue",
    "Grooming": "slategray",
    "Entry": "lime"
}


def reorder(tracedata):
    dflist = ['C24', 'C23', 'C20', 'C05', 'C06', 'C07', 'C00', 'C26', 'C22', 'C09',
              'C10', 'C17', 'C11', 'C21', 'C04', 'C01', 'C02', 'C03', 'C14', 'C15',
              'C16', 'C27', 'C25', 'C18', 'C19', 'C12', 'C13', 'C08']
    tracedata.reorder(dflist)


@utils.wrappers.log_time
def initialize_data(_filehandler: FileHandler, adjust: int = None):
    return CalciumData(_filehandler, color_dict, adjust=adjust)


def sparse_taste_data(ev_data):
    """Get specific subset of data based on particular events"""
    return TasteData(
        ev_data.tracedata, ev_data.time, ev_data.timestamps, ev_data.color_dict
    )


def statistics(_data, _dir) -> pd.DataFrame | None:
    return ProcessData(_data)


def pca(_pca):
    pca_plots = _pca.get_plots(colordict)
    pca_plots.scatterplot()


def plot(anal):
    df = anal.get_event_df()
    _pca = anal.get_pca(df)
    return _pca.get_plots(colordict)


def heatmap_loops(anal, cols):

    yield [heatmaps for heatmaps in anal.loop_taste(
        cols=cols,
        save_dir=r'C:\Users\dilorenzo\Desktop\CalciumPlots\heatmaps'
    )]

    # for hm in eatingdata.loop_eating(cols=dflist, save_dir=save_dir):
    #     my_hm = hm


def test(func):
    x = func
    lst = [j for j in x]
    return lst


if __name__ == "__main__":
    _animal = "PGT13"
    _date = "052622"
    _dir = r"C:\Users\flynn\repos\CalciumAnalysis\datasets"
    colordict = {'grooming': 'green',
                 'entry': 'blue',
                 'eating': 'red'}
    filehandler = FileHandler(
        _animal, _date, _dir, tracename="traces3", eatingname="Scored2"
    )
    data = initialize_data(filehandler, adjust=34)
    mypca = ca_pca.CaPrincipalComponentsAnalysis()
    graphs.plot.ScatterPlots(data.eatingdata.baseline())




