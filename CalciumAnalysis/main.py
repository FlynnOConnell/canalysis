# -*- coding: utf-8 -*-

"""
#main_nn.py

Module: Main code execution. 
        Note: Neural network requires separate main.py in neuralnetwork subpackage.
"""
from __future__ import annotations

import logging

from analysis.analysis_utils import ca_pca
from calcium_data import CalciumData
from data_utils.file_handler import FileHandler
from containers.taste_data import TasteData
from analysis.process_data import ProcessData
import pandas as pd
import faulthandler
import utils.funcs
from utils.wrappers import log_time
from graphs.plot import pca_scatter

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
    "Approach": "magenta",
    "Eating": "blue",
    "Grooming": "slategray",
    "Entry": "lime",
    "Interval": "grey"
}


def reorder(tracedata):
    dflist = ['C24', 'C23', 'C20', 'C05', 'C06', 'C07', 'C00', 'C26', 'C22', 'C09',
              'C10', 'C17', 'C11', 'C21', 'C04', 'C01', 'C02', 'C03', 'C14', 'C15',
              'C16', 'C27', 'C25', 'C18', 'C19', 'C12', 'C13', 'C08']
    tracedata.reorder(dflist)


@utils.wrappers.log_time
def initialize_data(_filehandler: FileHandler, adjust: int = None):
    return CalciumData(_filehandler, color_dict, adjust=adjust)


def statistics(_data, _dir) -> pd.DataFrame | None:
    return ProcessData(_data)


def heatmap_loops(_data, anal, cols):
    yield [heatmaps for heatmaps in anal.loop_taste(
        cols=cols,
        save_dir=r'C:\Users\dilorenzo\Desktop\CalciumPlots\heatmaps'
    )]
    for hm in _data.eatingdata.loop_eating(cols=cols, save_dir=save_dir):
        my_hm = hm


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
    mypca = ca_pca.get_pca(data.eatingdata.signals).pca_df
    scatter = pca_scatter(mypca, data.eatingdata.colors, color_dict=color_dict, edgecolors=None, s=6, spin=True)
    scatter[0].save()







