# -*- coding: utf-8 -*-
"""
#funcs.py

Module(util): General getter/setter/checker functions.
"""
from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
from glob import glob
from pathlib import Path
from typing import Tuple, Iterable, Optional, Sized

import matplotlib
import numpy as np
import pandas as pd

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


def remove_outliers(df,
                    colors,
                    std: Optional[int] = 2
                    ) -> Tuple[pd.DataFrame, pd.Series]:
    from scipy import stats

    df.reset_index(drop=True, inplace=True)
    colors.reset_index(drop=True, inplace=True)
    ind = (np.abs(stats.zscore(df)) < std).all(axis=1)
    df[ind] = df
    colors[ind] = colors
    assert colors.shape[0] == df.shape[0]

    return df, colors


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


def df_tolist(df) -> list:
    tmp = []   
    for col_name, ser in df.items(): 
        tmp.extend(ser)
    return tmp

def dict_df_tolist(dct, transpose: bool = False) - df: 
    for cell, df in dct.items():
        
    if transpose:
        
    
        


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
             gap: Optional[int] = 1,
             anti: Optional[bool] = False
             ) -> Iterable[list]:
    """
    Create intervals where there elements are separated by either:
        -less than gap.  (if anti = False (default))
        -more than gap.

    Args:
        lst (list): Iterable to search.
        gap (int): length of interval.
        anti (bool): Makes larger than (gap) intervals.
    Returns:
         interv (list): New list with created interval.
    """
    interv, tmp = [], []

    for v in lst:
        if not tmp:
            tmp.append(v)
        else:
            if anti:
                if abs(tmp[-1] - v) > gap:
                    tmp.append(v)
                elif abs(tmp[-1] - v) > gap:
                    tmp.append(v)
            else:
                interv.append([tmp[0], tmp[-1]])
                tmp = [v]

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
    logging.info('{} trace files found:'.format(len(files)))
    for file in files:
        logging.info(f'{file}')

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
    """
      Check iterable for duplicates.

      Args:
          to_check (Sized | Iterable[set]): Input iterable to check.

      Returns:
          Bool: Truth value if duplicates are present.

      """
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


def confidence_ellipse(x, y, ax, n_std=1.8, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
            
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    facecolor : str
        Color of background.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x 
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y 
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_handles(color_dict: dict,
                marker: Optional[str] = None,
                linestyle: Optional[int] = 'none',
                **kwargs
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
        proxy.append(matplotlib.lines.Line2D([0], [0], marker=marker, markerfacecolor=c, linestyle=linestyle, **kwargs))
        label.append(t)
    return proxy, label


# Main wrapper for testing
if __name__ == "__main__":
    pass
