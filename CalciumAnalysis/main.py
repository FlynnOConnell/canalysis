# -*- coding: utf-8 -*-

"""
#main_nn.py

Module: Main code execution. 
        Note: Neural network requires separate main.py in neuralnetwork subpackage.
"""
from __future__ import annotations

import logging
from data.calcium_data import CalciumData

from data.data_utils.file_handler import FileHandler
from data.taste_data import TasteData
from stats.process_data import ProcessData
from graphs.plot import Plot
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# %% Some examples on usage of different modules.

color_dict = {
    'ArtSal' : 'blue',
    'Citric' : 'yellow',
    'Lick'   : 'darkgray',
    'MSG'    : 'orange',
    'NaCl'   : 'green',
    'Quinine': 'red',
    'Rinse'  : 'lightsteelblue',
    'Sucrose': 'purple'
}


def initialize_data(_filehandler: FileHandler):
    _data = CalciumData(_filehandler, color_dict)
    return _data


def sparse_event_data(ev_data):
    """Get specific subset of data based on particular events"""
    taste_data = TasteData(ev_data.tracedata, ev_data.time, ev_data.timestamps,
                           ev_data.color_dict)
    return taste_data


def statistics(_data) -> pd.DataFrame | None:
    stats = ProcessData(_data)
    stats = stats.get_stats()
    return stats


if __name__ == "__main__":
    _animal = 'PGT13'
    _date = '052622'
    _dir = 'A:/'
    filehandler = FileHandler(_animal, _date, _dir)
    data = initialize_data(filehandler)

    graph = Plot(data.tracedata)

