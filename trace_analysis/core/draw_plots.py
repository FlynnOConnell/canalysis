#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# draw_plots.py

Module (core): Functions for drawing graphs.

"""
from __future__ import annotations
from typing import Optional

import pandas as pd
import numpy as np
import logging

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
from IPython.display import HTML
import webbrowser

from utils import funcs as func

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')


class Plot(object):
    def __init__(self, data, **kwargs):
        self.cmap = plt.get_cmap('viridis')
        self.data = data

    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None, caption: str = '', title=None, accuracy=None, legend_labels=None):
        x1 = X[:, 0]
        x2 = X[:, 1]
        class_distr = []

        y = np.array(y).astype(int)

        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        # Plot the different class distributions
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        # Plot legend
        if not legend_labels is None:
            plt.legend(class_distr, legend_labels, loc=1)

        if caption:
            plt.text(-.05, -.08,
                     caption,
                     fontstyle='italic',
                     fontsize='small')

        # Plot title
        if title:
            if accuracy:
                perc = 100 * accuracy
                plt.suptitle(title)
                plt.title(f"Accuracy: {perc}", fontsize=10)
            else:
                plt.title(title)

        # Axis labels
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.show()

    # Plot the dataset X and the corresponding labels y in 3D using PCA.
    def plot_in_3d(self, X, y=None):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
        plt.show()

    def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        ):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        from sklearn.model_selection import learning_curve

        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
        )
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(
            fit_time_sorted,
            test_scores_mean_sorted - test_scores_std_sorted,
            test_scores_mean_sorted + test_scores_std_sorted,
            alpha=0.1,
        )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt


    def confusion_matrix(y_test,
                         y_fit,
                         labels: list,
                         xaxislabel: Optional[str] = None,
                         yaxislabel: Optional[str] = None,
                         caption: Optional[str] = '',
                         save_dir: str = None) -> np.array:

        import seaborn as sns
        sns.set()
        from sklearn.metrics import confusion_matrix

        mat = confusion_matrix(y_test, y_fit)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=labels,
                    yticklabels=labels)

        if xaxislabel:
            plt.xlabel(xaxislabel)
        else:
            plt.xlabel('true label')
        if yaxislabel:
            plt.ylabel(yaxislabel)
        else:
            plt.ylabel('predicted label')
        if caption:
            plt.text(0, -.03,
                     caption,
                     fontstyle='italic',
                     fontsize='small')

        plt.show()

        if save_dir:
            plt.savefig(
                save_dir
                + '_confusionMatrix.png',
                Addbbox_inches='tight', dpi=300)

        return mat

    def scatter(df: pd.DataFrame,
                color_dict: dict,
                df_colors,
                three_dim: Optional[bool] = False,
                title: Optional[str] = None,
                plottype: Optional[str] = 'PCA',
                size: int = 5,
                marker: str = 'o',
                alpha: int = 1,
                save_dir: Optional[str] = None,
                msg: Optional[str] = None,
                caption: Optional[str] = None
                ) -> None:
        """
            Plot 2D/3D scatter plot with matplotlib.
    
            Args:
                df (pd.DataFrame): Input data for scatter plot. 
                color_dict (dict): Colors for graph.
                df_colors (pd.Series): Colors for each scatter point.
                three_dim (bool): Plot in 3D if True.
                title (str): Text to display as title.
                plottype (str): Type of graph to insert into title.
                size (int): Size of graph markers. Default = 5.
                marker (str): Shape of marker. Default is circle.
                alpha (int): Alpha of markers.
                save_dir (str): Optional alternative location to save.
                    Default = Path(resultsdir)
                msg (str): Message to include if save_dir is given.
                caption: Optional description box to place in graph.
    
            Returns:
                None
    
        """

        if title is None:
            title = 'Scatter Plot'

        fig = plt.figure()
        if three_dim:
            ax = fig.gca(projection='3d')
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.set_zlabel(df.columns[2])

            ax.scatter(df.PC1, df.PC2, df.PC3,
                       c=df.Color, s=size,
                       marker=marker, alpha=alpha)

        else:
            ax = fig.gca()
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.scatter(df.PC1, df.PC2,
                       c=df_colors, s=size,
                       marker=marker, alpha=alpha)

        ax.set_title('{}'.format(title) + ' ' + plottype)

        proxy, label = func.get_handles(color_dict)
        ax.legend(handles=proxy,
                  labels=label,
                  loc='upper right',
                  prop={'size': 6},
                  bbox_to_anchor=(1, 1),
                  ncol=2,
                  numpoints=1)

        if caption:
            fig.text(0, -.03,
                     caption,
                     fontstyle='italic',
                     fontsize='small')

        plt.show()
        if save_dir:
            if msg:
                msg = msg
            else:
                msg = 'No title given, include title in func param'
            fig.savefig(save_dir + '/{}_scatter.png'.format(msg),
                        bbox_inches='tight', dpi=300)

        return None

    def plot_skree(var: np.ndarray) -> None:
        lab = np.arange(len(var)) + 1
        plt.plot(lab, var, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained (%)')
        leg = plt.legend(['Eigenvalues from SVD'],
                         loc='best',
                         borderpad=0.3,
                         shadow=False,
                         prop=fm.FontProperties(size='small'),
                         markerscale=0.4)

        leg.get_frame().set_alpha(0.4)
        leg.set_draggable(state=True)
        plt.show()

    def plot_3d_ani(session: str,
                    df: pd.DataFrame,
                    color_dict: dict,
                    size: int = 5,
                    marker: str = 'o',
                    alpha: int = 1,
                    save_dir: Optional[str] = None) -> None:
        """
            Animated 3D Scatter plot.
            Plot gets saved to a temporary html file (location provided by save_dir).
    
            Args:
                session (str): Animal and date of recording.
                df (DataFrame): Scaled, normalized matrix with loading scores.
                color_dict (dict): Colors for graph.
                size (int): Size of graph markers. Default = 5.
                marker (str): Shape of marker. Default is circle.
                alpha (int): Alpha of markers.
                save_dir (str): Optional alternative location to save. Default = Path(resultsdir)
    
            Returns:
                None
    
        """

        fig = plt.figure()

        ax = fig.gca(projection='3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('{}'.format(session) + ' ' + '3D Scatter PCA')
        ax.scatter(df.PC1, df.PC2, df.PC3, c=df.Color, s=size, marker=marker, alpha=alpha)

        proxy, label = func.get_handles(color_dict)

        ax.legend(handles=proxy,
                  labels=label,
                  loc='upper right',
                  prop={'size': 6},
                  bbox_to_anchor=(1, 1),
                  ncol=2,
                  numpoints=1)

        def init():
            ax.plot(df.PC1, df.PC2, df.PC3, linewidth=0, antialiased=False)
            return fig,

        def animate(i):
            ax.view_init(elev=30., azim=3.6 * i)
            return fig,

        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=400, interval=100, blit=True)

        data = HTML(ani.to_html5_video())
        with open(r'C:/Users/dilorenzo/Desktop.htm', 'wb') as f:
            f.write(data.data.encode("UTF-8"))

        if save_dir:
            url = save_dir
        else:
            url = r'C:/Users/dilorenzo/Desktop.htm'
        webbrowser.open(url, new=2)

        return None

    def plot_session(nplot: int,
                     tracedata: pd.DataFrame,
                     time: pd.Series,
                     session: str,
                     numlicks: int,
                     timestamps: dict = None,
                     lickshade: int = 1,

                     save_dir: str = None) -> None:
        # create a series of plots with a shared x-axis
        fig, axs = plt.subplots(nplot, 1, sharex=True)
        for i in range(nplot):
            # get calcium trace (y axis data)
            signal = list(tracedata.iloc[:, i + 1])

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
        if save_dir:
            fig.savefig(save_dir / '/{}_session.png'.format(session),
                        bbox_inches='tight', dpi=300)
            logger.info(msg=f'File saved to {save_dir}')

        return None

    def plot_zoom(nplot,
                  tracedata,
                  time,
                  timestamps,
                  session,
                  colors,
                  zoomshade: float = 0.2,
                  save_dir: Optional[str] = None
                  ) -> None:
        # create a series of plots with a shared x-axis
        fig, ax = plt.subplots(nplot, 1, sharex=True)
        zoombounding = [
            int(input('Enter start time for zoomed in graph (seconds):')),
            int(input('Enter end time for zoomed in graph (seconds):'))
        ]

        for i in range(nplot):
            signal = list(tracedata.iloc[:, i + 1])

            # plot signal
            ax[i].plot(time, signal, 'k', linewidth=.8)
            ax[i].get_xaxis().set_visible(False)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].set_yticks([])
            ax[i].set_ylabel(tracedata.columns[i + 1],
                             rotation='horizontal', labelpad=15, y=.1)

            # go through each set of timestamps and shade them accordingly
            for stim, times in timestamps.items():
                for ts in times:
                    if ts == times[0]:
                        label = stim
                    else:
                        label = '_'  # Keeps label from showing.
                    ax[i].axvspan(ts, ts + zoomshade,
                                  color=colors[stim], label=label, lw=0)

        # Make the plots act like they know each other
        fig.subplots_adjust(hspace=0)
        plt.xlabel('Time (s)')
        fig.suptitle('Calcium Traces: {}'.format(session), y=.95)

        ax[-1].get_xaxis().set_visible(True)
        ax[-1].spines["bottom"].set_visible(True)

        # set the x-axis to the zoomed area
        plt.setp(ax, xlim=zoombounding)
        plt.legend(loc=(1.02, 3))

        if save_dir:
            fig.savefig(save_dir / '/{}_zoom.png'.format(session),
                        bbox_inches='tight', dpi=300)
            logger.info(msg=f'File saved to {save_dir}')

        return None

    def plot_stim(nplot,
                  tracedata,
                  time,
                  timestamps,
                  trial_times,
                  session,
                  colors,
                  save_dir: str = None
                  ) -> None:
        for stim, times in trial_times.items():
            trialno = 0
            for trial in times:
                trialno += 1

                # get only the data within the analysis window
                data_ind = np.where((time > trial - 4) & (time < trial + 5))[0]
                # index to analysis data
                this_time = time[data_ind]

                fig, ax = plt.subplots(nplot, 1, sharex=True)
                for i in range(nplot):
                    # Get calcium trace for this analysis window
                    signal = list(tracedata.iloc[data_ind, i + 1])
                    # plot signal
                    ax[i].plot(this_time, signal, 'k', linewidth=1)
                    ax[i].get_xaxis().set_visible(False)
                    ax[i].spines["top"].set_visible(False)
                    ax[i].spines["bottom"].set_visible(False)
                    ax[i].spines["right"].set_visible(False)
                    ax[i].set_yticks([])
                    ax[i].set_ylabel(tracedata.columns[i + 1],
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
                                ax[i].axvspan(
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
                ax[-1].get_xaxis().set_visible(True)
                ax[-1].spines["bottom"].set_visible(True)
                fig.set_figwidth(4)

        if save_dir:
            fig.savefig(save_dir / '/{}_session.png'.format(session),
                        bbox_inches='tight', dpi=300)
            logger.info(msg=f'File saved to {save_dir}')

        return None

    def plot_cells(tracedata,
                   time,
                   timestamps,
                   trial_times,
                   session,
                   colors,
                   save_dir: str = None
                   ) -> None:
        cells = tracedata.columns[1:]
        plot_cells = func.cell_gui(tracedata)

        for cell in plot_cells:

            # Create int used to index a column in tracedata.
            cell_index = {x: 0 for x in cells}
            cell_index.update((key, value) for value, key in enumerate(cell_index))

            currcell = cell_index[cell]
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
                    cell, session, stim), y=1.0)
                fig.set_figwidth(6)
                fig.text(0, -.03,
                         ('Note: Each trace has'
                          'been normalized over the graph window'),
                         fontstyle='italic', fontsize='small')
                xaxs[-1, 0].spines["bottom"].set_bounds(False)
                plt.legend(loc=(1.02, 3))

        if save_dir:
            fig.savefig(save_dir / '/{}_cell.png'.format(session),
                        bbox_inches='tight', dpi=300)

        return None
