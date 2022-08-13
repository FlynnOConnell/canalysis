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
    "ArtSal": "dodgerblue",
    "Peanut": "darkorange",
    "Lick": "darkgray",
    "Chocolate": "saddlebrown",
    "NaCl": "green",
    "Quinine": "red",
    "Acid": "yellow",
    "Sucrose": "purple",
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
    stats = stats.get_stats()
    return stats


if __name__ == "__main__":
    _animal = "PGT13"
    _date = "052622"
    _dir = "A:/"

    filehandler = FileHandler(
        _animal, _date, _dir, tracename="traces2", eatingname="Scored1"
    )
    data = initialize_data(filehandler, adjust=34)

    myarr = np.array(data.eventdata.timestamps["Lick"])

    # graph = Plot(data.tracedata)
    # analysis = ProcessData(data, _dir)
    # for plot in analysis.loop_taste():
    #     myplot = plot

    # pca = analysis.PCA(data.tracedata.signals)
    # data.plot_session()
    # data.plot_zoom(save_dir=save_dir)
