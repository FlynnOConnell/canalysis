#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:28:23 2022

@author: flynnoconnell
"""

import sys
import easygui
import Analysis.Func as func
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as ss
import matplotlib as plt
import matplotlib.lines
import matplotlib.animation as animation
from IPython.display import HTML
import webbrowser


def get_colors(colors_fill, allcells):
    pca_colors = []

    for i in range(0, len(allcells)):
        if len(allcells) > len(colors_fill):
            print("You need to add more colors, too many cells not enough colors")
            sys.exit()
        else:
            i += 1
            pca_colors.append(colors_fill[i])
    return pca_colors


def get_pca(pca_df, pca):
    scaled_df = ss().fit_transform(pca_df)
    pca.fit(scaled_df)  # loading scores and variation
    transform_df = pca.transform(scaled_df)  # final transform

    return transform_df


def pca_cells(transform_df, cell_names, per_var):
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    pca_df_c = pd.DataFrame(transform_df, index=cell_names, columns=labels)

    return pca_df_c, per_var, labels


def pca_time(trc, time, labels):
    scaled_trc = ss().fit_transform(trc)
    pca = PCA()
    pca.fit(scaled_trc)  # calc loading scores and variation
    pcaT_trc = pca.transform(scaled_trc)  # final transform
    perT_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labelsT = ['PC' + str(x) for x in range(1, len(perT_var) + 1)]
    pcaT_df = pd.DataFrame(pcaT_trc, index=time, columns=labels)

    return pcaT_df, perT_var, labelsT


def scatter(x, y, z, label_x, label_y, label_z, session, pca_df_c):
    """  Make scatter plot, given data and labels.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)
    ax.set_title('{}'.format(session) + ' ' + '3D Scatter PCA')
    ax.scatter(x, y, z, s=40, c=pca_df_c.colors, marker='o', alpha=1)

    handles = [
        matplotlib.lines.Line2D(
            [], [], marker="o", color=c,
            linestyle="none") for c in pca_df_c.colors.values]
    ax.legend(handles=handles, labels=list(pca_df_c.index.values),
              loc='upper right', prop={'size': 6},
              bbox_to_anchor=(1, 1), ncol=2, numpoints=1)
    return fig


def scatter_t(x, y, z, label_x, label_y, label_z, session, test_dir):
    """  Make scatter plot, given data and labels.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)
    ax.set_title('{}'.format(session) + ' ' + '3D Scatter PCA')
    ax.scatter(x, y, z, s=5, marker='o', alpha=1)
    ax.plot(x, y, z)
    fig.savefig(test_dir + '/{}_session.png'.format(session),
                bbox_inches='tight', dpi=300)
    # handles = [matplotlib.lines.Line2D([],[], marker="o", color=c, linestyle="none") for c in pca_df.Licks.values]
    # ax.legend(handles=handles, labels=list(pca_dfT.columns.values),
    #            loc='upper right', prop={'size':6}, bbox_to_anchor=(1,1),ncol=2, numpoints=1)
    return fig


def scatter_ani(x, y, z, label_x, label_y, label_z, session, pca_df):
    """ Make animated scatter plot, given data and labels.
        Opens scatter plot in default browser.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)
    ax.set_title('{}'.format(session) + ' ' + '3D Scatter PCA')
    ax.scatter(x, y, z, s=5, c=pca_df.Licks, marker='o', alpha=1)

    # handles = [matplotlib.lines.Line2D([],[], marker="o", color=c, linestyle="none") for c in pca_df.colors.values]
    # plt.legend(handles=handles, labels=list(pca_df.index.values),
    #            loc='upper right', prop={'size':6}, bbox_to_anchor=(1,1),ncol=2, numpoints=1)

    def init():
        ax.plot(x, y, z, linewidth=0, antialiased=False)
        return fig,

    def animate(i):
        ax.view_init(elev=30., azim=3.6 * i)
        return fig,

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=400, interval=100, blit=True)

    data = HTML(ani.to_html5_video())
    with open('C:\\Temp\\3d_plot.htm', 'wb') as f:
        f.write(data.data.encode("UTF-8"))

    url = r'C:\\Temp\\3d_plot.htm'
    webbrowser.open(url, new=2)

    return fig


def plot_cell(allcells, singlecells):
    plotcells = []

    while True:
        myplotcells = easygui.multchoicebox(
            msg='Please select cell(s) to plot, or cancel to continue.',
            choices=allcells)
        if myplotcells is None:  # If no cells chosen
            ynbox = easygui.ynbox(
                msg=('No cells chosen, continue without'
                     'plotting individual cells?'),
                choices=(
                    ['Yes', 'No, take me back']))
            if ynbox is False:
                continue
            if ynbox is True:
                singlecells = 0
                break
        else:
            plotcells.append(myplotcells)
            break

    return plotcells, singlecells

# TODO: Merge functions with similar parameters


def line_session(nplot, tracedata, time, timestamps, lickshade, session, numlicks):
    # create a series of plots with a shared x-axis
    fig, axs = plt.subplots(nplot, 1, sharex=True)
    for i in range(nplot):
        # get calcium trace (y axis data)
        signal = list(tracedata.iloc[:, i + 1])
        signal = func.get_signal(i, tracedata)

        # plot signal
        axs[i].plot(time, signal, 'k', linewidth=.8)

        # Get rid of borders
        axs[i].get_xaxis().set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["bottom"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].set_yticks([])  # no Y ticks

        # add the cell name as a label for this graph's y-axis
        axs[i].set_ylabel(tracedata.columns[i + 1],
                          rotation='horizontal', labelpad=15, y=.1)

        # go through each lick and shade it in
        for lick in timestamps['Lick']:
            label = '_yarp'
            if lick == timestamps['Lick'][0]:
                label = 'Licking'
            axs[i].axvspan(lick, lick + lickshade,
                           color='lightsteelblue', lw=0, label=label)

    # Make the plots act like they know each other
    fig.subplots_adjust(hspace=0)
    plt.xlabel('Time (s)')
    fig.suptitle('Calcium Traces: {}'.format(session), y=.95)
    plt.legend(loc=(.98, 7))
    axs[-1].get_xaxis().set_visible(True)
    axs[-1].spines["bottom"].set_visible(True)
    fig.text(0, -.03, 'Number of Licks: {}'.format(numlicks),
             fontstyle='italic', fontsize='small')

    return fig


def line_zoom(nplot, tracedata, time, timestamps, session, zoomshade, colors):
    # create a series of plots with a shared x-axis
    zoomfig, zaxs = plt.subplots(nplot, 1, sharex=True)
    zoombounding = [
        int(input('Enter start time for zoomed in graph (seconds):')),
        int(input('Enter end time for zoomed in graph (seconds):'))
    ]

    for i in range(nplot):
        signal = list(tracedata.iloc[:, i + 1])

        # plot signal
        zaxs[i].plot(time, signal, 'k', linewidth=.8)
        zaxs[i].get_xaxis().set_visible(False)
        zaxs[i].spines["top"].set_visible(False)
        zaxs[i].spines["bottom"].set_visible(False)
        zaxs[i].spines["right"].set_visible(False)
        zaxs[i].set_yticks([])
        zaxs[i].set_ylabel(tracedata.columns[i + 1],
                           rotation='horizontal', labelpad=15, y=.1)
        # go through each set of timestamps and shade them accordingly
        for stim, times in timestamps.items():
            for ts in times:
                if ts == times[0]:
                    label = stim
                else:
                    label = '_'  # Keeps label from showing.
                zaxs[i].axvspan(ts, ts + zoomshade,
                                color=colors[stim], label=label, lw=0)

    # Make the plots act like they know each other
    zoomfig.subplots_adjust(hspace=0)
    plt.xlabel('Time (s)')
    zoomfig.suptitle('Calcium Traces: {}'.format(session), y=.95)

    zaxs[-1].get_xaxis().set_visible(True)
    zaxs[-1].spines["bottom"].set_visible(True)

    # set the x-axis to the zoomed area
    plt.setp(zaxs, xlim=zoombounding)
    plt.legend(loc=(1.02, 3))

    return zoomfig


def line_stim(nplot, tracedata, time, timestamps, trial_times, session, colors):
    for stim, times in trial_times.items():
        trialno = 0
        for trial in times:
            trialno += 1

            # get only the data within the analysis window
            data_ind = np.where((time > trial - 4) & (time < trial + 5))[0]
            # index to analysis data
            this_time = time[data_ind]

            fig, axs = plt.subplots(nplot, 1, sharex=True)
            for i in range(nplot):
                # Get calcium trace for this analysis window
                signal = list(tracedata.iloc[data_ind, i + 1])
                # plot signal
                axs[i].plot(this_time, signal, 'k', linewidth=1)
                axs[i].get_xaxis().set_visible(False)
                axs[i].spines["top"].set_visible(False)
                axs[i].spines["bottom"].set_visible(False)
                axs[i].spines["right"].set_visible(False)
                axs[i].set_yticks([])
                axs[i].set_ylabel(tracedata.columns[i + 1],
                                  rotation='horizontal',
                                  labelpad=15, y=.1)
                # Add shading.
                for stimmy in ['Lick', 'Rinse', stim]:
                    done = 0
                    timey = timestamps[stimmy]
                    for ts in timey:
                        if trial - 4 < ts < trial + 5:
                            if done == 0:
                                label = stimmy
                                done = 1
                            else:
                                label = '_'
                            axs[i].axvspan(
                                ts, ts + .15,
                                color=colors[stimmy],
                                label=label, lw=0)

            # Make the plots act like they know each other.
            fig.subplots_adjust(hspace=0)
            plt.xlabel('Time (s)')
            fig.suptitle('Calcium Traces\n{}: {} trial {}'.format(
                session, stim, trialno), y=1.0)
            fig.text(0, -.03,
                     ('Note: Each trace'
                      'normalized over the graph window'),
                     fontstyle='italic', fontsize='small')
            axs[-1].get_xaxis().set_visible(True)
            axs[-1].spines["bottom"].set_visible(True)
            fig.set_figwidth(4)

    return fig, stim, trialno


def line_single(
        my_cell, cell_index, tracedata, time,
        timestamps, trial_times, session, colors):
    # Create int used to index a column in tracedata.
    currcell = cell_index[my_cell]

    for stim, times in trial_times.items():
        ntrial = len(times)
        minmax = []
        # Max/mins to standardize plots
        for it, tri in enumerate(times):
            temp_data_ind = np.where(
                (time > tri - 2) & (time < tri + 5))[0]
            temp_signal = tracedata.iloc[temp_data_ind, currcell + 1]
            norm_min = min(temp_signal)
            norm_max = max(temp_signal)
            minmax.append(norm_min)
            minmax.append(norm_max)
        stim_min = min(minmax)
        stim_max = max(minmax)

        if ntrial > 1:
            fig, xaxs = plt.subplots(
                ntrial, 1, sharex=False, squeeze=False)
        if ntrial == 1:
            xaxs.flatten()
        for iteration, trial in enumerate(times):
            i = int(iteration)

            data_ind = np.where(
                (time > trial - 2) & (time < trial + 4))[0]
            this_time = time[data_ind]

            # Get calcium trace.
            signal = list(tracedata.iloc[data_ind, currcell + 1])
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
                timey = timestamps[stimmy]
                for ts in timey:
                    if trial - 1.5 < ts < trial + 5:
                        if done == 0:
                            label = stimmy
                            done = 1
                        else:
                            label = '_'
                        xaxs[i, 0].axvspan(
                            ts, ts + .15,
                            color=colors[stimmy],
                            label=label, lw=0)

        plt.xlabel('Time (s)')
        fig.suptitle('Calcium Traces: {}\n{}: {}'.format(
            my_cell, session, stim), y=1.0)
        fig.set_figwidth(6)
        fig.text(0, -.03,
                 ('Note: Each trace has'
                  'been normalized over the graph window'),
                 fontstyle='italic', fontsize='small')
        xaxs[-1, 0].spines["bottom"].set_bounds(False)
        plt.legend(loc=(1.02, 3))

        return fig
