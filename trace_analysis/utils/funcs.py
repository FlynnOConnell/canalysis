# -*- coding: utf-8 -*-
"""
#funcs.py

Module(util): General getter/setter/checker functions.
"""
from __future__ import annotations
from typing import Tuple, Iterable, Optional, Sized
import os

import matplotlib
import pandas as pd
import numpy as np
import math
import contextlib
from pathlib import Path
from glob import glob
import gc
import logging

from utils import excepts as e

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


@contextlib.contextmanager
def working_directory(path: str) -> None:
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
    return None


def garbage_collect(*delvar) -> None:
    """Clear global variables in current namespace."""

    for _ in delvar:
        del _
    gc.collect()
    logging.info('Recycled unused variables.')

    return None


def clear_all() -> None:
    """
    Clears all the variables from the workspace.
    """
    gl = globals().copy()
    for var in gl:
        if var[0] == '_':
            continue
        if 'func' in str(globals()[var]):
            continue
        if 'module' in str(globals()[var]):
            continue

        del globals()[var]

    return None


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def flatten(lst: list) -> list:
    return [item for sublist in lst for item in sublist]


def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def count_unique(arr: Iterable[any], n: int) -> int:
    """
    Count number of unique values in an array.
    Args:
        arr (np.ndarray): 1d array to count.
        n (int): how many unique values to count.
    Returns:
        res (int): number of unique values in given array.
    """
    res = 1

    # Pick all elements one by one
    for i in range(1, n):
        j = 0
        for j in range(i):
            if arr[i] == arr[j]:
                break

        # If not printed earlier, then print it
        if i == j + 1:
            res += 1

    return res


def get_unique(arr: Iterable[any]) -> list:
    """
    Return all unique elements in an array.
    Args:
        arr (np.ndarray): 1d array to search.
    Returns:
         is_unique (list): List of unique elements
    """
    is_unique = []
    for x in arr:
        # check if exists in unique_list or not
        if x not in is_unique:
            is_unique.append(x)

    return is_unique


def interval(lst: Iterable[any],
             gap: int = 1) -> Iterable[list]:
    """
    Create intervals where there elements are separated by less than 1.
    Used for bout creation.
    
    Args:
        lst (list): Iterable to search.
    Returns:
         interv (list): New list with created interval.
    """
    interv, tmp = [], []
    
    for v in lst:
        if not tmp:
            tmp.append(v)
        else:
            if abs(tmp[-1] - v) < gap:
                tmp.append(v)
            else:
                interv.append([tmp[0], tmp[-1]])
                tmp = [v]
                
    # res = [[ele for ele in sub if ele[0] == ele[1]] for sub in interv]
                
    return interv


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
    logging.info('{} trace files found:'.format(len(files)))
    
    if len(files) > 1:
        if pick == 0: 
            tracepath = Path(files[0])
            logging.info('Taking trace file: {}'.format(tracepath.name))
        else: 
            tracepath = Path(files[pick])
            logging.info('Taking trace file: {}'.format(tracepath.name))
    elif len(files) == 1: 
        tracepath = Path(files[0])
        
    eventpath = Path(glob(os.path.join(datapath, '*processed*'))[0])
    tracedata = pd.read_csv(tracepath, low_memory=False)
    eventdata = pd.read_csv(eventpath, low_memory=False)

    return tracedata, eventdata


def dup_check(signal: list | np.ndarray,
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
    '''
      Check iterable for duplicates.
      
      Args:
          to_check (Sized | Iterable[set]): Input iterable to check.
      
      Returns:
          Bool: Truth value if duplicates are present.
      
      '''
    return len(to_check) != len(set(to_check))


def uniquify(path: str) -> str:
    """ Make unique filename if path already exists.
    """
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path


def get_peak_window(time: list | pd.Series, peak_time) -> list:
    """
    Returns the index of tracedata centered 1s around the peak flourescent value for that trial.
    Args:
        time (list | pd.Series): List of all time values.
        peak_time (float): Time value where signal is at its largest value.
    Returns:
         window_ind (list): list of index values to match time.
    """
    time: list
    peak_time: float

    aux, window_ind = [], []
    for valor in time:
        aux.append(abs(peak_time - valor))

    window_ind.append(aux.index(min(aux)) - 20)
    window_ind.append(aux.index(min(aux)) + 20)

    return window_ind


def get_matched_time(time: Iterable[any],
                     match: list,
                     return_index: Optional[bool] = False) -> list:
    """
    Finds the closest number in tracedata time to the input. Can be a single value, or list.

    :param time: Series
    :param argv: list
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


def get_handles(color_dict: dict,
                linestyle: Optional[str] = 'none',
                marker: str = 'o'
                ) -> Tuple[list, list]:
    """
    Get matplotlib handles for input dictionary.

    Args:
        color_dict (dict): Dictionary of event:color k/v pairs.
        linestyle (str): Connecting lines, default = none.
        marker (str): Shape of scatter point, default is circle.

    Returns:
        proxy (list): matplotlib.lines.line2D appended list.
        label (list): legend labels for each proxy.

    """

    proxy, label = [], []
    for t, c in color_dict.items():
        proxy.append(matplotlib.lines.Line2D([0], [0], linestyle=linestyle, c=c, marker=marker))
        label.append(t)

    return proxy, label


# Main wrapper for testing
if __name__ == "__main__":
    pass
