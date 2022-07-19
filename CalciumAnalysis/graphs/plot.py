#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# draw_plots.py

Module (core): Functions for drawing graphs.

"""
from __future__ import annotations

import logging
import webbrowser
from typing import Optional, Iterable, Any

import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML
from data.calcium_data import CalciumData
from matplotlib import rcParams

import graphs.graph_utils.graph_funcs as gr_func
from data.trace_data import TraceData


def set_pub():
    # Function to set some easy params and avoid some annoying bugs
    rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": 'bold',
        'axes.facecolor': 'w',
        "axes.labelsize": 15,
        "lines.linewidth": 1,
    })


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')


class Plot(object):

    def __init__(self,
                 data: pd.DataFrame | Any = None,
                 colors: Optional[Iterable] = None,
                 cmap: str = 'magma',
                 dpi: Optional[int] = 600,
                 save_dir: str = None,
                 **kwargs
                 ):
        """
        Class with graphing utilities. 
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Plotting data, representing points on the graph. The default is None.
        colors : Optional[Iterable], optional
            Matched colors if data values are colored. 
            colors.shape must be the same as data.shape[0]. The default is None.
        cmap : str, optional
            Color for colormap. The default is 'magma'.
        dpi : Optional[int], optional
            Pixel resolution. The default is 600.
        save_dir : str, FileHandler, optional
            Directory for saving data. The default is None.
        **kwargs : dict
            optional arguments for matplotlib.Axes and Axes.Scatter.

        Returns
        -------
        None.

        """
        self.data = data
        self.colors = colors
        self.dpi = dpi
        self.save_dir = save_dir
        self.facecolor = 'white'
        self.color_dict = None,
        self.cmap = plt.get_cmap(cmap)

        self.kwargs = kwargs
        self.checks = {}

    def scatter(self,
                df: pd.DataFrame = None,
                ax=None,
                colors: Iterable | Any = None,
                title: Optional[str] = None,
                legend: Optional[bool] = True,
                size: int = 5,
                marker: str = 'o',
                alpha: Optional[int] = 1,
                conf_interv: Optional[bool] = False,
                savefig: Optional[bool] = False,
                msg: Optional[str] = 'nomsg',
                bbox_inches: str = 'tight',
                facecolor: str = 'white',
                **kwargs,
                ) -> None:
        """
            Plot 2D/3D scatter plot with matplotlib.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Input data for scatter plot. The default is None.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        colors : Iterable | Any, optional
            Colors for graph. The default is None.
        title : Optional[str], optional
            Text to display as title. The default is None.
        legend : Optional[bool], optional
            Whether to include a legend. The default is True.
        size : int, optional
            Size of graph markers. Default = 5. The default is 5.
        marker : str, optional
            Shape of marker. The default is 'o'.
        alpha : Optional[int], optional
            Alpha of markers. The default is 1.
        conf_interv : Optional[bool], optional
            Whether to plot ellipse confidence intervals.
            The default is False.
        savefig : Optional[bool], optional
            Alternative location to save. The default is None.
        msg : Optional[str], optional
            Message to include if save_dir is given. The default is 'nomsg'.
        bbox_inches : str, optional
            Whitespace surrounding graphs. The default is 'tight'.
        facecolor : str, optional
            Color of background. The default is 'white'.
        **kwargs : dict
            Optional inputs to matplotlib.Axes.Scatter.

        Returns
        -------
        None

        """
        set_pub()

        if 'colors' in df.columns:
            colors = df.pop('colors')

        fig = plt.figure()
        fig.set_dpi(self.dpi)

        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        ax = ax or fig.gca()

        ax.set_xlabel(df.columns[0], weight='bold')
        ax.set_ylabel(df.columns[1], weight='bold')
        ax.patch.set_facecolor = 'white'
        ax.scatter(x, y,
                   c=colors,
                   s=size,
                   marker=marker,
                   alpha=alpha,
                   facecolor=facecolor,
                   **kwargs
                   )

        df.reset_index(drop=True, inplace=True)
        colors.reset_index(drop=True, inplace=True)

        if conf_interv:
            for color in np.unique(colors):
                _df = df.loc[(df['colors'] == color)]
                gr_func.confidence_ellipse(
                    _df.iloc[:, 0],
                    _df.iloc[:, 1],
                    ax,
                    facecolor=color,
                    edgecolor='k',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.08)
        if title:
            ax.set_title(f'{title}', fontweight='bold')
        if legend:
            proxy, label = gr_func.get_handles(self.color_dict, marker='o', markersize=size, alpha=1)
            ax.legend(handles=proxy,
                      labels=label,
                      loc='lower right',
                      prop={'size': 6},
                      bbox_to_anchor=(1.05, 1),
                      ncol=2,
                      numpoints=1)

        plt.show()
        if savefig:
            fig.savefig(self.save_dir + f'/{title}_{msg}.png',
                        bbox_inches=bbox_inches, dpi=self.dpi, facecolor=self.facecolor)
        return None

    @staticmethod
    def skree(variance: np.ndarray,
              title: str = '') -> None:
        """
        Line chart skree plot.

        Parameters
        ----------
        variance : np.ndarray
            From PCA.explained_variance_ratio_.
        title : str, optional
            Title of graph. The default is ''.

        Returns
        -------
        None
            DESCRIPTION.
        """
        lab = np.arange(len(variance)) + 1
        plt.plot(lab, variance, 'o-', linewidth=2, color='blue')
        plt.title(f'{title}' + 'Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained (%)')
        leg = plt.legend(['Eigenvalues from SVD'],
                         loc='best',
                         borderpad=0.3,
                         shadow=False,
                         prop=fm.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(1)
        plt.show()

        return None

    def scatter_3d(self,
                   df: pd.DataFrame = None,
                   color_dict: dict = None,
                   color: pd.Series = None,
                   title: Optional[str] = None,
                   size: int = 5,
                   marker: str = 'o',
                   alpha: int = 1,
                   conf_interv: bool = True,
                   savefig: Optional[bool] = False,
                   msg: Optional[str] = None,
                   caption: Optional[str] = None,
                   dpi: int = 500,
                   bbox_inches: str = 'tight',
                   facecolor: str = 'white'
                   ) -> None:
        """
        Plot 3D scatter plot with matplotlib.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Input data for scatter plot. The default is None.
        color_dict : dict, optional
            Colors for graph. The default is None.
        color : pd.Series, optional
            Colors for each scatter point. The default is None.
        title : Optional[str], optional
            Text to display as title. The default is None.
        size : int, optional
            Size of graph markers. The default is 5.
        marker : str, optional
            Shape of marker. The default is 'o'.
        alpha : int, optional
            Alpha of markers. The default is 1.
        conf_interv : Optional[bool], optional
            Whether to plot ellipse confidence intervals.
            The default is False.
        savefig : Optional[bool], optional
            Alternative location to save. The default is None.
        msg : Optional[str], optional
            Message to include if save_dir is given. The default is 'nomsg'.
        bbox_inches : str, optional
            Whitespace surrounding graphs. The default is 'tight'.
        facecolor : str, optional
            Color of background. The default is 'white'.

        Returns
        -------
        None
            DESCRIPTION.
        """

        assert isinstance(df, pd.DataFrame)

        if 'colors' in df.columns:
            self.colors = df.pop('colors')

        color = self.colors
        fig = plt.figure()
        fig.set_dpi(300)

        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        z = df.iloc[:, 2]

        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(df.columns[0], weight='bold')
        ax.set_ylabel(df.columns[1], weight='bold')
        ax.set_zlabel(df.columns[2], weight='bold')

        ax.patch.set_facecolor = 'white'

        ax.scatter(x,
                   y,
                   z,
                   c=color,
                   s=size,
                   marker=marker,
                   alpha=alpha,
                   facecolor='white'
                   )

        df.reset_index(drop=True, inplace=True)
        color.reset_index(drop=True, inplace=True)

        if conf_interv:
            for color in np.unique(color):
                _df = df.loc[(df['colors'] == color)]

                gr_func.confidence_ellipse(
                    _df.iloc[:, 0],
                    _df.iloc[:, 1],
                    ax,
                    facecolor=color,
                    edgecolor='k',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.08)

        ax.set_title(f'{title}', fontweight='bold')

        proxy, label = gr_func.get_handles(color_dict, marker='o', markersize=5, alpha=1)
        ax.grid(linewidth=.1)
        ax.legend(handles=proxy,
                  labels=label,
                  loc='lower right',
                  prop={'size': 6},
                  bbox_to_anchor=(1.05, 1),
                  ncol=2,
                  numpoints=1)

        if caption:
            fig.text(0, -.03,
                     caption,
                     fontstyle='italic',
                     fontsize='small')

        plt.show()
        if savefig:
            fig.savefig(self.save_dir + f'/{msg}_{title}.png',
                        bbox_inches=bbox_inches, dpi=500, facecolor=self.facecolor)
            logging.info(f'Fig saved to {self.save_dir}: {bbox_inches}, {dpi}, {facecolor}')

        return None

    def plot_3d_ani(self,
                    session: str,
                    df: pd.DataFrame,
                    color_dict: dict,
                    size: int = 5,
                    marker: str = 'o',
                    alpha: int = 1
                    ) -> None:
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

            Returns:
                None
    
        """
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        z = df.iloc[:, 2]

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('{}'.format(session) + ' ' + 'PCA')
        ax.scatter(x, y, z, c=self.colors, s=size, marker=marker, alpha=alpha)

        proxy, label = gr_func.get_handles(color_dict)
        ax.legend(handles=proxy,
                  labels=label,
                  loc='upper right',
                  prop={'size': 6},
                  bbox_to_anchor=(1, 1),
                  ncol=2,
                  numpoints=1)

        def init():
            ax.plot(x, y, z, linewidth=0, antialiased=False)
            return fig,

        def animate(i):
            ax.view_init(elev=30., azim=3.6 * i)
            return fig,

        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=400, interval=100, blit=True)

        data = HTML(ani.to_html5_video())
        with open('/Users/flynnoconnell/Pictures', 'wb') as f:
            f.write(data.data.encode("UTF-8"))

        url = self.save_dir
        webbrowser.open(url, new=2)

        return None

    @staticmethod
    def confusion_matrix(
                         y_pred,
                         y_true,
                         labels: list,
                         xaxislabel: Optional[str] = None,
                         yaxislabel: Optional[str] = None,
                         caption: Optional[str] = '') -> np.array:
        """
        

        Parameters
        ----------
        y_pred : TYPE
            DESCRIPTION.
        y_true : TYPE
            DESCRIPTION.
        labels : list
            DESCRIPTION.
        xaxislabel : Optional[str], optional
            DESCRIPTION. The default is None.
        yaxislabel : Optional[str], optional
            DESCRIPTION. The default is None.
        caption : Optional[str], optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        mat : TYPE
            DESCRIPTION.

        """

        import seaborn as sns
        sns.set()
        from sklearn.metrics import confusion_matrix
        set_pub()
        mat = confusion_matrix(y_pred, y_true)
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
                     fontsize='small')
        plt.show()
        return mat
