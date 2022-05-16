# -*- coding: utf-8 -*-

"""
#main_nn.py

Module: Main code execution. 
        Note: Neural network requires separate main.py in neuralnetwork subpackage.
"""
import logging
from data.calcium_data import CalciumData
from parameters.data_params import *
from file_handling.file_handler import FileHandler
from taste_data import TasteData

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# %% Some examples on usage of different modules.

def initialize_data(_filehandler: FileHandler):
    _data = CalciumData(_filehandler)
    return _data


def sparse_event_data(ev_data):
    """Get specific subset of data based on particular events"""
    taste_data = TasteData(ev_data.tracedata, ev_data.time, ev_data.timestamps, ev_data.color_dict)
    return taste_data


if __name__ == "__main__":
    _animal = data_params.config['SINGLE']['animal']
    _date = data_params.config['SINGLE']['date']
    _dir = data_params.config['DIRS']['HOME']
    filehandler = FileHandler(_animal, _date, _dir)
    data = initialize_data(filehandler)
