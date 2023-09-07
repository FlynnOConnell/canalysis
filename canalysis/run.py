from __future__ import annotations

import os
from inspect import getsourcefile
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any


from canalysis.data.containers import CalciumData
from canalysis.data.data_utils.file_handler import FileHandler
import yaml

__location__ = os.path.abspath(getsourcefile(lambda: 0))
ROOT_DIR = os.path.dirname(__location__)

_hard_dependencies = ["numpy", "pandas"]  # let users know if theyre missing any vital deps
_missing_dependencies = []

for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:
        _missing_dependencies.append(f"{_dependency}: {_e}")

if _missing_dependencies:
    raise ImportError("Unable to import required dependencies:\n" + "\n".join(_missing_dependencies))
del _hard_dependencies, _dependency, _missing_dependencies


class Params:
    """@DynamicAttrs"""

    def __init__(self, parameter_dict):
        for key in parameter_dict:
            setattr(self, key, parameter_dict[key])


def get_parameters():
    with open(os.path.join(ROOT_DIR, "params.yaml"), "rb") as f:
        parameters = yaml.safe_load(f.read())
    return Params(parameters)


def get_data():
    params = get_parameters()
    filehandler = FileHandler(
        params.Session["animal"],
        params.Session["date"],
        params.Directory["data"],
        params.Filenames["traces"],
        params.Filenames["events"],
        params.Filenames["gpio"],
        params.Filenames["eating"],
    )
    return CalciumData(
        filehandler,
        doeating=params.Filenames["doeating"],
        doevents=params.Filenames["doevents"],
        color_dict=params.Colors,
        adjust=params.Filenames["adjust"],
    )


def plot_traces(trialtimes: Dict[str, List[float]], df: pd.DataFrame, signal_time: np.array):

    window_post_event = 10  # 4 seconds after the event

    for event, times in trialtimes.items():
        if event in ["Quinine", "Sucrose", "MSG", "Citric"]:
            for idx, cell in enumerate(df.columns):
                if idx >= 2:  # Plot only first 5 cells for debugging
                    break
                plt.figure()
                plt.title(f"{cell} for {event} event")

                all_trials = []  # To hold data for calculating the average

                for time in times:
                    start_time = time
                    end_time = time + window_post_event

                    # Find the index in signal_time for start_time and end_time
                    idx = np.where((signal_time >= start_time) & (signal_time <= end_time))[0]
                    timeidx = signal_time[idx]

                    # Slice the DataFrame to get the signal 4 seconds post each event
                    sliced_data = df.iloc[idx][cell]

                    # Re-index to start at 0
                    sliced_data.index = timeidx - start_time

                    # Append to all_trials for calculating the average later
                    all_trials.append(sliced_data)
                    fig, ax = plt.subplots()
                    ax.set_title(f"{cell} for {event} event")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Signal")
                    ax.plot(sliced_data.index, sliced_data.values, label=f"Trial at {time}s")
                    plt.show()
                    x = 2

                # Calculate the average signal across all trials for the cell
                avg_signal = pd.concat(all_trials, axis=1).mean(axis=1)
                plt.plot(avg_signal.index, avg_signal.values, label="Average", linewidth=2, color="black")

                plt.legend()
                plt.xlabel("Time (s)")
                plt.ylabel("Signal")
                plt.show()


if __name__ == "__main__":
    from pathlib import Path
    import seaborn as sns

    data = get_data()
    savename = Path().home() / "Dropbox" / "Lab"
    trials = data.tastedata.trial_times
    signals = data.tracedata.signals
    signal_time = data.tracedata.time

    sns.set_style("darkgrid")
    sns.set_palette("husl")

    cell_to_plot = "C01"  # The cell you're interested in

    for event, times in trials.items():
        if event == "Rinse":
            continue
        # Initialize a list to hold the sliced data for this cell for each time
        cell_data_list = []

        # Create figure and axis objects
        fig, ax = plt.subplots()
        plt.title(f"Cell {cell_to_plot} for event {event}", fontsize=16)

        for time in times:
            start_time = time - 2  # 2 seconds before the event
            end_time = time + 4  # 4 seconds after the event

            # Get the indices corresponding to this time range
            idx = np.where((signal_time >= start_time) & (signal_time <= end_time))[0]

            # Extract the relevant signal slice for this cell and time
            sliced_data = signals.iloc[idx][cell_to_plot]

            # Check if data is as expected
            if sliced_data.shape[0] != len(idx):
                print(f"Skipping time {time} for event {event} due to mismatch in data.")
                continue

            # Store it in the list
            cell_data_list.append(sliced_data.values)

        # Average across all times for this cell and event
        avg_signal = np.mean(cell_data_list, axis=0)

        # Generate the time axis corresponding to -2 to +4 seconds
        time_axis = np.linspace(-2, 4, len(avg_signal))

        sns.lineplot(x=time_axis, y=avg_signal, ax=ax, label=f"{event} average", linewidth=2)

        ax.legend(fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("DF/F", fontsize=14)
        ax.tick_params(labelsize=12)

        sns.despine()

        plt.show()

    # data.plot_session(save=False)
    # data.plot_zoom(save=False, savename=savename / "zoom", cells=data.cells[:5].tolist(), zoombounding=(1000, 1400))
    # corr = data.tracedata.signals.corr()
    # # Generate a mask for the upper triangle, excluding the diagonal
    # mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # k=1 excludes the diagonal
    #
    # # Set up the matplotlib figure
    # fig, ax = plt.subplots(figsize=(10, 8))
    #
    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 20, as_cmap=True)
    #
    # # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    #
    # plt.title("Cell Correlation Heatmap")
    # plt.show()
    # # Calculate correlations
    # pre = data.tastedata.tastedata.drop(columns="color")
    # # One-hot encode the events
    # one_hot_events = pd.get_dummies(pre["event"], prefix="event")
    # neural_data_one_hot = pd.concat([pre, one_hot_events], axis=1).drop("event", axis=1)
    # # Calculate the correlation matrix
    # correlation_matrix = neural_data_one_hot.corr()
    #
    # # Isolate the subset of the correlation matrix that describes the correlation between neurons and events
    # cell_names = data.tracedata.cells  # Replace this with the actual list of your cell names
    # event_names = one_hot_events.columns  # This will dynamically get all the event names
    #
    # # Get the sub-matrix that contains only the correlations between the specified neurons and events
    # neuron_event_corr = correlation_matrix.loc[cell_names, event_names]
    #
    # # Plot this subset as a heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(neuron_event_corr, annot=False, cmap="coolwarm", cbar=True, square=False)
    # plt.title("Correlation between Neurons and Events")
    # plt.show()
