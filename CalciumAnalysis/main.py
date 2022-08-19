# -*- coding: utf-8 -*-

"""
#main_nn.py

Module: Main code execution. 
        Note: Neural network requires separate main.py in neuralnetwork subpackage.
"""
from __future__ import annotations

import logging

import numpy as np

from data.calcium_data import CalciumData

from data.data_utils.file_handler import FileHandler
from data.taste_data import TasteData
from analysis.process_data import ProcessData
from graphs.plot import Plot
import pandas as pd
import faulthandler

faulthandler.enable()


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

save_dir = "C:/Users/dilorenzo/Desktop/CalciumPlots/"
# %% Some examples on usage of different modules.

color_dict = {
    "ArtSal"   : "dodgerblue",
    "Peanut"   : "darkorange",
    "Lick"     : "darkgray",
    "Chocolate": "saddlebrown",
    "NaCl"     : "green",
    "Quinine"  : "red",
    "Acid"     : "yellow",
    "Sucrose"  : "purple",
}


def initialize_data(_filehandler: FileHandler, adjust: int = None):
    _data = CalciumData(_filehandler, color_dict, adjust=adjust)
    return _data


def sparse_event_data(ev_data):
    """Get specific subset of data based on particular events"""
    taste_data = TasteData(
        ev_data.tracedata, ev_data.time, ev_data.timestamps, ev_data.color_dict
    )
    return taste_data


def statistics(_data, _dir) -> pd.DataFrame | None:
    stats = ProcessData(_data, outpath=_dir)
    return stats


if __name__ == "__main__":
    _animal = "PGT13"
    _date = "052622"
    _dir = r"C:\Users\dilorenzo\Documents\repos\CalciumAnalysis\datasets"

    filehandler = FileHandler(
        _animal, _date, _dir, tracename="traces3", eatingname="Scored1"
    )
    data = initialize_data(filehandler, adjust=34)

    analysis = ProcessData(data, _dir)

    dflist = ['C24', 'C23', 'C20', 'C05', 'C06', 'C07', 'C00', 'C26', 'C22', 'C09',
              'C10', 'C17', 'C11', 'C21', 'C04', 'C01', 'C02', 'C03', 'C14', 'C15',
              'C16', 'C27', 'C25', 'C18', 'C19', 'C12', 'C13', 'C08']
    # df = analysis.get_event_df()
    # df, var = analysis.principal_components(data)
    plots_ = [heatmaps for heatmaps in analysis.loop_taste(
        cols=dflist,
        save_dir=r'C:\Users\dilorenzo\Desktop\CalciumPlots\heatmaps'
    )]

    plots = [heatmaps for heatmaps in analysis.loop_eating(
        cols=dflist,
        save_dir=r'C:\Users\dilorenzo\Desktop\CalciumPlots\heatmaps'
    )]
    # save_dir='C:/Users/flynn/Desktop/figs')]


