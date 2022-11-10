"""
A package for analysis, statistics and visualization of Calcium Imaging data,
specifically from including from Inscopix.
"""
from __future__ import annotations
from inspect import getsourcefile
import os
import logging
import yaml
from canalysis.data.containers import CalciumData
from canalysis.data.data_utils.file_handler import FileHandler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
__location__ = os.path.abspath(getsourcefile(lambda:0))
ROOT_DIR = os.path.dirname(__location__)

class Params:
    """@DynamicAttrs"""
    def __init__(self, parameter_dict):
        for key in parameter_dict:
            setattr(self, key, parameter_dict[key])


def get_parameters():
    with open(os.path.join(ROOT_DIR, 'params.yaml'),
              'rb') as f:
        parameters = yaml.safe_load(f.read())
    return Params(parameters)

def get_data(doeating: bool = True, doevents: bool = True, adjust=None):
    params = get_parameters()
    filehandler = FileHandler(
            params.Session['animal'],
            params.Session['date'],
            params.Directory['data'],
            params.Filenames['traces'],
            params.Filenames['events'],
            params.Filenames['gpio'],
            params.Filenames['eating'],
    )
    return CalciumData(filehandler,
                       doeating=params.Filenames['doeating'],
                       doevents=params.Filenames['doevents'],
                       color_dict=params.Colors,
                       adjust=params.Filenames['adjust'],
                       )
    
# Path: CalciumAnalysis/CalciumAnalysis/data/containers/calcium_data.py
__all__ = [
    "CalciumData",
    "FileHandler",
    "get_data",
    "Params",
    "Plot",
    "Heatmaps",
    "ProcessData",
    ]