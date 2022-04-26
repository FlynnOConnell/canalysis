#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#_plots.py

Module (graph): .Mixin functions to inherit, plotting functions that use instance attributes from Calcium Data
 class and nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as plt
from typing import Optional, Iterable


class CalPlots:

    signals: pd.DataFrame | pd.Series | np.ndarray | Iterable
    time: np.ndarray | Iterable | int | float | bool
    cells: np.ndarray | pd.Series | Iterable | int | float | bool
    tracedata: pd.DataFrame
    session: str
    trial_times: dict
    timestamps: dict
    color_dict: dict

    def plot_stim(self,
                  save_dir: str = None
                  ) -> None:

        if save_dir:
            save_dir = save_dir
        else:
            save_dir = '/Users/flynnoconnell/Pictures/sessions2'

        nplot = len(self.signals.columns)
        for stim, times in self.trial_times.items():
            trialno = 0
            for trial in times:
                trialno += 1

                # get only the data within the analysis window
                data_ind = np.where((self.time > trial - 2) & (self.time < trial + 5))[0]
                # index to analysis data
                this_time = self.time[data_ind]

                fig, ax = plt.subplots(nplot, 1, sharex=True)
                for i in range(0, nplot):
                    # Get calcium trace for this analysis window
                    signal = list(self.signals.iloc[data_ind, i])
                    # plot signal
                    ax[i].plot(this_time, signal, 'k', linewidth=1)
                    ax[i].get_xaxis().set_visible(False)
                    ax[i].spines["top"].set_visible(False)
                    ax[i].spines["bottom"].set_visible(False)
                    ax[i].spines["right"].set_visible(False)
                    ax[i].set_yticks([])
                    ax[i].set_ylabel(self.signals.columns[i],
                                     rotation='horizontal',
                                     labelpad=15, y=.1)
                    # Add shading.
                    for stimmy in ['Lick', 'Rinse', stim]:
                        done = 0
                        timey = self.timestamps[stimmy]
                        for ts in timey:
                            if trial - 1 < ts < trial + 3:
                                if done == 0:
                                    label = stimmy
                                    done = 1
                                else:
                                    label = '_'
                                ax[i].axvspan(
                                    ts, ts + .15,
                                    color=self.color_dict[stimmy],
                                    label=label, lw=0)

                # Make the plots act like they know each other.
                fig.subplots_adjust(hspace=0)
                plt.xlabel('Time (s)')
                ax[-1].get_xaxis().set_visible(True)
                ax[-1].spines["bottom"].set_visible(True)
                fig.set_figwidth(4)

                if save_dir:
                    fig.savefig(save_dir + f'/{stim}_{trial}.png',
                                bbox_inches='tight', dpi=600, facecolor='white')

        return None

    def plot_session(self,
                     lickshade: int = 1,
                     save_dir: bool = True) -> None:

        # create a series of plots with a shared x-axis
        fig, axs = plt.subplots(len(self.cells), 1, sharex=True, facecolor='white')
        for i in range(len(self.cells)):
            # get calcium trace (y axis data)
            signal = list(self.signals.iloc[:, i])

            # plot signal
            axs[i].plot(self.time, signal, 'k', linewidth=.8)

            # Get rid of borders
            axs[i].get_xaxis().set_visible(False)
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].set_yticks([])  # no Y ticks

            # add the cell name as a label for this graph's y-axis
            axs[i].set_ylabel(self.signals.columns[i],
                              rotation='horizontal', labelpad=15, y=.1, fontweight='bold')

            # go through each lick and shade it in
            for lick in self.timestamps['Lick']:
                label = '_yarp'
                if lick == self.timestamps['Lick'][0]:
                    label = 'Licking'
                axs[i].axvspan(lick, lick + lickshade,
                               color='lightsteelblue', lw=0, label=label)

        # Make the plots act like they know each other
        fig.subplots_adjust(hspace=0)
        plt.xlabel('Time (s)')
        fig.suptitle(f'Calcium Traces: {self.session}', y=.95)
        # plt.legend(loc=(1.04, 1))
        axs[-1].get_xaxis().set_visible(True)
        axs[-1].spines["bottom"].set_visible(True)
        if save_dir:
            fig.savefig(f'/Users/flynnoconnell/Pictures/plots/{self.session}_zm.png',
                        bbox_inches='tight', dpi=600, facecolor='white')

        return None

    def plot_zoom(self,
                  zoomshade: float = 0.2,
                  save_dir: Optional[bool] = True
                  ) -> None:
        # create a series of plots with a shared x-axis
        fig, ax = plt.subplots(len(self.cells), 1, sharex=True)
        zoombounding = [
            int(input('Enter start time for zoomed in graph (seconds):')),
            int(input('Enter end time for zoomed in graph (seconds):'))
        ]

        for i in range(len(self.cells)):
            signal = list(self.signals.iloc[:, i])

            # plot signal
            ax[i].plot(self.time, signal, 'k', linewidth=.8)
            ax[i].get_xaxis().set_visible(False)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].set_yticks([])
            ax[i].set_ylabel(self.signals.columns[i],
                             rotation='horizontal', labelpad=15, y=.1)

            # go through each set of timestamps and shade them accordingly
            for stim, times in self.timestamps.items():
                for ts in times:
                    if ts == times[0]:
                        label = stim
                    else:
                        label = '_'  # Keeps label from showing.
                    ax[i].axvspan(ts, ts + zoomshade,
                                  color=self.color_dict[stim], label=label, lw=0)

        # Make the plots act like they know each other
        fig.subplots_adjust(hspace=0)
        plt.xlabel('Time (s)')
        fig.suptitle(f'Calcium Traces: {self.session}', y=.95)

        ax[-1].get_xaxis().set_visible(True)
        ax[-1].spines["bottom"].set_visible(True)

        # set the x-axis to the zoomed area
        plt.setp(ax, xlim=zoombounding)
        plt.legend(loc=(1.02, 3))

        if save_dir:
            fig.savefig(f'/Users/flynnoconnell/Pictures/plots/{self.session}_zm.png',
                        bbox_inches='tight', dpi=600)
        return None

    def plot_cells(self,
                   save_dir: Optional[bool] = True
                   ) -> None:
        cells = self.cells
        if cells is None:
            cells = self.cells

        for cell in cells:

            # Create int used to index a column in tracedata.
            cell_index = {x: 0 for x in cells}
            cell_index.update((key, value) for value, key in enumerate(cell_index))

            currcell = cell_index[cell]
            for stim, times in self.trial_times.items():
                ntrial = len(times)
                minmax = []
                # Max/mins to standardize plots
                for it, tri in enumerate(times):
                    temp_data_ind = np.where(
                        (self.time > tri - 2) & (self.time < tri + 5))[0]
                    temp_signal = self.tracedata.iloc[temp_data_ind, currcell + 1]
                    norm_min = min(temp_signal)
                    norm_max = max(temp_signal)
                    minmax.append(norm_min)
                    minmax.append(norm_max)
                stim_min = min(minmax)
                stim_max = max(minmax)


                fig, xaxs = plt.subplots(
                    ntrial, 1, sharex=False, squeeze=False)

                if ntrial < 2:
                    xaxs.flatten()

                for iteration, trial in enumerate(times):
                    i = int(iteration)

                    data_ind = np.where(
                        (self.time > trial - 2) & (self.time < trial + 4))[0]
                    this_time = self.time[data_ind]

                    # Get calcium trace.
                    signal = list(self.tracedata.iloc[data_ind, currcell + 1])
                    signal[:] = [number - stim_min for number in signal]

                    l_bound = min(signal)
                    u_bound = max(signal)
                    center = 0
                    xaxs[i, 0].plot(this_time, signal, 'k', linewidth=.8)
                    xaxs[i, 0].tick_params(
                        axis='both', which='minor', labelsize=6)

                    xaxs[i, 0].get_xaxis().set_visible(False)

                    xaxs[i, 0].spines["top"].set_visible(False)
                    xaxs[i, 0].spines["bottom"].set_visible(False)
                    xaxs[i, 0].spines["right"].set_visible(False)
                    xaxs[i, 0].spines['left'].set_bounds(
                        (l_bound, stim_max))

                    xaxs[i, 0].set_yticks((0, center, u_bound))
                    xaxs[i, 0].set_ylabel(' Trial {}     '.format(
                        i + 1), rotation='horizontal', labelpad=15, y=.3)

                    xaxs[i, 0].set_ylim(bottom=0, top=max(signal))

                    xaxs[i, 0].axhspan(0, 0, color='k', ls=':')

                    # Add shading for licks, rinses  tastant delivery
                    for stimmy in ['Lick', 'Rinse', stim]:
                        done = 0
                        timey = self.timestamps[stimmy]
                        for ts in timey:
                            if trial - 1.5 < ts < trial + 5:
                                if done == 0:
                                    label = stimmy
                                    done = 1
                                else:
                                    label = '_'
                                xaxs[i, 0].axvspan(
                                    ts, ts + .15,
                                    color=self.color_dict[stimmy],
                                    label=label, lw=0)

                plt.xlabel('Time (s)')
                fig.suptitle('Calcium Traces: {}\n{}: {}'.format(
                    cell, self.session, stim), y=1.0)
                fig.set_figwidth(6)
                fig.text(0, -.03,
                         ('Note: Each trace has'
                          'been normalized over the graph window'),
                         fontstyle='italic', fontsize='small')
                xaxs[-1, 0].spines["bottom"].set_bounds(False)
                plt.legend(loc=(1.02, 3))

        if save_dir:
            plt.savefig(str(save_dir) + '.png',
                        bbox_inches='tight', dpi=600)

        return None
