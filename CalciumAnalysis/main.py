# -*- coding: utf-8 -*-
"""
#main.py

Module: Main code execution. 
        Note: Neural network requires separate main.py in neuralnetwork subpackage.
"""

from __future__ import annotations
from CalciumAnalysis import get_data
import pandas as np

data = get_data(doevents=False, adjust=120.3)


