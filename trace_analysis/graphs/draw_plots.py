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
from matplotlib import rcParams

import graphs.graph_funcs as gr_func


def set_pub():
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
                 data: pd.DataFrame=None,
                 legend=None,
                 colors: Optional[Iterable] = None,
                 events: Optional[Iterable] = None,
                 cmap: str ='magma',
                 dpi:  Optional[int] = 600,
                 save_dir: str = '/Users/flynnoconnell/Pictures',
                 **kwargs
                 ):

        self.cmap = plt.get_cmap(cmap)
        self.data = data
        self.colors = colors
        self.dpi = dpi
        self.save_dir = save_dir
        
        self.facecolor = 'white'
        self.color_dict = {
            'ArtSal': 'dodgerblue',
            'MSG': 'darkorange',
            'NaCl': 'lime',
            'Sucrose': 'magenta',
            'Citric': 'yellow',
            'Quinine': 'red',
            'Rinse': 'lightsteelblue',
            'Lick': 'darkgray'
        }
        
        self.kwargs = kwargs
        self.checks = {}

    def scatter(self,
                df: pd.DataFrame = None,
                ax=None,
                colors: Iterable | Any = None,
                title: Optional[str] = None,
                legend: Optional[bool] = True,
                ax_labels: Optional[bool] = True,
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
    
            Args:
                df (pd.DataFrame): Input data for scatter plot. 
                color_dict (dict): Colors for graph.
                colors_c (pd.Series) | Any: Colors for each scatter point.
                title (str): Text to display as title.
                legend (bool: Whether to include a legend
                conf_interv (bool): Whether to plot ellipse confidence intervals.
                size (int): Size of graph markers. Default = 5.
                marker (str): Shape of marker. Default is circle.
                alpha (int): Alpha of markers.
                savefig (str): Optional alternative location to save.
                    Default = Path(resultsdir)
                msg (str): Message to include if save_dir is given.
                caption: Optional description box to place in graph.
                facecolor (str): Color of background.
                bbox_inches (str): Whitespace surrounding graphs.
                dpi (int): pixels per inch.

            Returns:
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
    def skree(var: np.ndarray,
                   title: str = '') -> None:

        lab = np.arange(len(var)) + 1
        plt.plot(lab, var, 'o-', linewidth=2, color='blue')
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

    def scatter_2d(self,
                   df: pd.DataFrame | Iterable[Any],
                   y=None,
                   lim: bool = True,
                   label_x=None,
                   label_y=None,
                   label_z=None,
                   colors: Optional[Iterable] = None,
                   legend: Optional[bool] = True,
                   color_dict=None,
                   marker: Optional[str] = 'o',
                   title: Optional[str] = ''
                   ):
        """ 
        Make scatter plot, given data and labels.  
        """
        if color_dict is None:
            color_dict = self.legend
        if isinstance(df, pd.DataFrame):
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            # label_x = df.columns[0]
            # label_y = df.columns[1]
        else:
            x = df
            y = y
            
        colors = colors
        fig = plt.figure()
        ax = fig.gca()
        
        if lim: 
            ax.set_xlim(-.5, .5)
            ax.set_ylim(0.5, 5)
        # ax.set_xlabel(label_x, weight='bold')
        # ax.set_ylabel(label_y, weight='bold')
        # ax.set_title(f'{title}')
        ax.scatter(x, y, s=40, c=colors, marker=marker, alpha=1)
        if legend:
            proxy, labels = gr_func.get_handles(color_dict)
            ax.legend(handles=proxy,
                      labels=labels,
                      loc='upper right',
                      prop={'size': 6},
                      bbox_to_anchor=(1, 1),
                      ncol=2, numpoints=1)

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
            Plot 2D/3D scatter plot with matplotlib.
    
            Args:
                df (pd.DataFrame): Input data for scatter plot. 
                color_dict (dict): Colors for graph.
                colors_c (pd.Series) | (np.1darray): Colors for each scatter point.
                title (str): Text to display as title.
                size (int): Size of graph markers. Default = 5.
                marker (str): Shape of marker. Default is circle.
                alpha (int): Alpha of markers.
                savefig (str): Optional alternative location to save.
                    Default = Path(resultsdir)
                msg (str): Message to include if save_dir is given.
                caption: Optional description box to place in graph.
    
            Returns:
                None
    
        """

        assert isinstance(df, pd.DataFrame)
        
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

    def plot_3d_ani(session: str,
                    df: pd.DataFrame,
                    color_dict: dict,
                    colors: pd.Series,
                    size: int = 5,
                    marker: str = 'o',
                    alpha: int = 1,
                    temp_dir: Optional[str] = None) -> None:
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
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        z = df.iloc[:, 2]

        fig = plt.figure()

        ax = fig.gca(projection='3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('{}'.format(session) + ' ' + 'PCA')
        ax.scatter(x, y, z, c=colors, s=size, marker=marker, alpha=alpha)

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

        if temp_dir:
            url = temp_dir
        else:
            url = '/Users/flynnoconnell/Pictures'
        webbrowser.open(url, new=2)

        return None

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
