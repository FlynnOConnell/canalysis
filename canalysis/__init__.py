"""
A package for analysis, statistics and visualization of Calcium Imaging data,
specifically from including from Inscopix.
"""
from __future__ import annotations

import logging
import os
from inspect import getsourcefile

import yaml

from canalysis.data.containers import CalciumData
from canalysis.data.data_utils.file_handler import FileHandler

__location__ = os.path.abspath(getsourcefile(lambda:0))
ROOT_DIR = os.path.dirname(__location__)

_hard_dependencies = ["numpy", "pandas"] # let users know if theyre missing any vital deps
_missing_dependencies = []

for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:
        _missing_dependencies.append(f"{_dependency}: {_e}")

if _missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _hard_dependencies, _dependency, _missing_dependencies

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

def get_data():
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

  