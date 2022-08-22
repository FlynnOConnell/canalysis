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
from graphs.plot import ScatterPlots
from analysis.analysis_utils.analysis_funcs import map_colors
from analysis.analysis_utils import ca_pca
import pandas as pd
import faulthandler
from analysis.analysis_utils.analysis_funcs import map_colors
faulthandler.enable()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

save_dir = "C:/Users/dilorenzo/Desktop/CalciumPlots/"
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

dflist = ['C24', 'C23', 'C20', 'C05', 'C06', 'C07', 'C00', 'C26', 'C22', 'C09',
          'C10', 'C17', 'C11', 'C21', 'C04', 'C01', 'C02', 'C03', 'C14', 'C15',
          'C16', 'C27', 'C25', 'C18', 'C19', 'C12', 'C13', 'C08']


def initialize_data(_filehandler: FileHandler, adjust: int = None):
    return CalciumData(_filehandler, color_dict, adjust=adjust)


def sparse_taste_data(ev_data):
    """Get specific subset of data based on particular events"""
    return TasteData(
        ev_data.tracedata, ev_data.time, ev_data.timestamps, ev_data.color_dict
    )


def statistics(_data, _dir) -> pd.DataFrame | None:
    return ProcessData(_data, outpath=_dir)


def pca(_pca):
    pca_plots = _pca.get_plots(colordict)
    pca_plots.scatterplot()


def plot(anal):
    df = anal.get_event_df()
    _pca = anal.get_pca()
    return _pca.get_plots(colordict)


def heatmap_loops(anal):
    yield [heatmaps for heatmaps in anal.loop_taste(
        cols=dflist,
        save_dir=r'C:\Users\dilorenzo\Desktop\CalciumPlots\heatmaps'
    )], [heatmaps for heatmaps in anal.loop_eating(
        cols=dflist,
        save_dir=r'C:\Users\dilorenzo\Desktop\CalciumPlots\heatmaps'
    )]


if __name__ == "__main__":
    _animal = "PGT13"
    _date = "052622"
    _dir = r"C:\Users\dilorenzo\Documents\repos\CalciumAnalysis\datasets"
    colordict = {'grooming': 'green',
                 'entry': 'blue',
                 'eating': 'red'}
    filehandler = FileHandler(
        _animal, _date, _dir, tracename="traces3", eatingname="Scored1"
    )
    data = initialize_data(filehandler, adjust=34)
    tastedata = data.tastedata
    eatingdata = data.eatingdata
    # analysis = ProcessData(data)
    # eating_data = analysis.get_event_df()
    # eating_data['colors'] = map_colors(eating_data.pop('events'))
    # for event, taste_df in tastedata.taste_events.items():
    #     eating_df = eating_data.copy().reset_index(drop=True)
    #     data = pd.concat([eating_df, taste_df], axis=1)
    #     colors = data.pop('colors')
    #     pca_class = ca_pca.CaPrincipalComponentsAnalysis(data, colors)
    #     plot = ScatterPlots(pca_class.data, pca_class.colors, colordict)



    # pca = analysis_class.get_pca()
    # pca_plots = pca.get_plots(colordict)
    # data = pca.pca_df
    # data_colors = pca.colors

    # for a, b in tastedata_class.generate_stim_data():
    #     temp_cols = data_colors.copy()
    #     data_ = pd.concat([data, b])
    #     temp_cols.extend(data_colors)



   # def get_pca(self):
   #      data = self.get_event_df()
   #      data_events = data.pop('events')
   #      data_colors = map_colors(data_events)
   #      pca = ca_pca.CaPrincipalComponentsAnalysis(data=data, colors=data_colors,)
   #      return pca