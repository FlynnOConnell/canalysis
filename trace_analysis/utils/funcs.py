# -*- coding: utf-8 -*-
"""
#funcs.py

Module(util): General getter/setter/checker functions.
"""
from __future__ import annotations

import math
import os
from glob import glob
from pathlib import Path
from typing import Tuple, Iterable, Optional, Sized, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import logging
from utils import excepts as e
from utils.wrappers import typecheck

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


# %% COLLAPSE DATA STRUCTURES

def keys_exist(element, *keys):
    """
    Check if *keys (nested) exists in `element` (dict).
    """

    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


@typecheck(dict, int)
def iter_events(event_dct, gap: int = 5):
    for event, ts in event_dct.items():
        intervals = interval(ts, gap)
        for interv in intervals:
            yield event, interv


@typecheck(Iterable)
def flatten(lst: Iterable) -> list:
    return [item for sublist in lst for item in sublist]


@typecheck(Iterable, int)
def interval(lst: Iterable[any],
             gap: Optional[int] = 1,
             outer: bool = False
             ) -> list[tuple[Any, Any]]:
    """
    Create intervals where there elements are separated by either:
        -less than gap. 
        -more than gap.

    Args:
        lst (list): Iterable to search.
        gap (int): length of interval.
        outer (bool): Makes larger than (gap) intervals.
    Returns:
         interv (list): New list with created interval.
    """
    interv, tmp = [], []

    for v in lst:
        if not tmp:
            tmp.append(v)

        elif abs(tmp[-1] - v) < gap:
            tmp.append(v)

        elif outer:
            interv.append(tuple((tmp[-1], v)))
            tmp = [v]
        else:
            interv.append(tuple((tmp[0], tmp[-1])))
            tmp = [v]
    return interv


@typecheck(pd.DataFrame, pd.Series, int)
def remove_outliers(df,
                    colors,
                    std: Optional[int] = 2
                    ) -> Tuple[pd.DataFrame, pd.Series]:
    df.reset_index(drop=True, inplace=True)
    colors.reset_index(drop=True, inplace=True)
    ind = (np.abs(stats.zscore(df)) < std).all(axis=1)
    df[ind] = df
    colors[ind] = colors
    assert colors.shape[0] == df.shape[0]

    return df, colors


def get_dir(data_dir: str,
            _id: str,
            date: str,
            pick: int
            ):
    """
    
    From Home directory, set path to data files with pathlib object variable.
    
    Directory structure: 
        -| Animal 
        --| Date
        ---| Results
        -----| Graphs
        -----| Statistics
        ---| Data_traces*
        ---| Data_gpio_processed*

    Args:
        data_dir (str): Path to directory
        pick (int) = index of files to choose.
        _id (str): Current animal ID
        date (str): Current session date  
        
    Returns:
        tracedata (pd.DataFrame): DataFrame of cell signals
        eventdata (pd.DataFrame): DataFrame of event times.
        session (str): Concatenated name of animalID_Date
        
    """

    os.chdir(data_dir)
    datapath = Path(data_dir) / _id / date

    files = (glob(os.path.join(datapath, '*traces*')))
    logging.info(f'{len(files)} trace files found:')
    for file in files:
        file = Path(file)
        logging.info(f'{file.stem}')
        logging.info('-' * 15)

    if len(files) > 1:
        if pick == 0:
            tracepath = Path(files[0])
            logging.info('Taking trace file: {}'.format(tracepath.name))
        else:
            tracepath = Path(files[pick])
            logging.info('Taking trace file: {}'.format(tracepath.name))
    elif len(files) == 1:
        tracepath = Path(files[0])
    else:
        logging.info(f'Files for {_id}, {date} not found.')
        raise FileNotFoundError

    eventpath = Path(glob(os.path.join(datapath, '*processed*'))[0])
    tracedata = pd.read_csv(tracepath, low_memory=False)
    eventdata = pd.read_csv(eventpath, low_memory=False)

    return tracedata, eventdata


## Data Validation


@typecheck(Iterable[any])
def dup_check(signal: Iterable[any],
              peak_signal: float | int) -> None:
    """
    Check if multiple peak signals are present in taste-response moving window. 

    Args:
        signal (list | np.ndarray): Signal to validate.
        peak_signal (float | int): Largest value in moving window of taste responses.

    Raises:
        exception: DuplicateError.

    Returns:
        None.

    """
    checker = []
    for value in signal:
        dupcheck = math.isclose(value, peak_signal, abs_tol=1.0)
        if dupcheck is True:
            checker.append(dupcheck)
    if not checker:
        raise e.DuplicateError()
    return None


def has_duplicates(to_check: Sized | Iterable[set]):
    """
      Check iterable for duplicates.

      Args:
          to_check (Sized | Iterable[set]): Input iterable to check.

      Returns:
          Bool: Truth value if duplicates are present.

      """
    return len(to_check) != len(set(to_check))


@typecheck(Iterable[any], Iterable[any])
def get_peak_window(time: Iterable[any], peak: float) -> list:
    """
    Returns the index of tracedata centered 1s around the peak flourescent value for that trial.
    Args:
        time (list | pd.Series): List of all time values.
        peak (float) : peak time
    Returns:
         window_ind (list): list of index values to match time.
    """
    time: list

    aux, window_ind = [], []
    for valor in time:
        aux.append(abs(peak - valor))

    window_ind.append(aux.index(min(aux)) - 20)
    window_ind.append(aux.index(min(aux)) + 20)

    return window_ind


# @typecheck(Iterable[any], Iterable[any], bool)
def get_matched_time(time: Iterable[any],
                     match: Iterable[any],
                     return_index: Optional[bool] = False) -> list:
    """
    Finds the closest number in tracedata time to the input. Can be a single value, or list.

    :param time: Series
    :param match: list
    :param return_index: bool
    """
    time: pd.Series

    matched_index = []
    matched_time = []
    for t in match:
        temp = []
        for valor in time:
            temp.append(abs(t - valor))
        matched_index.append(temp.index(min(temp)))

    for idx in matched_index:
        this_time = time[idx]
        matched_time.append(this_time)

    if return_index:
        return matched_index
    else:
        return matched_time


# Main wrapper for testing
if __name__ == "__main__":
    pass
