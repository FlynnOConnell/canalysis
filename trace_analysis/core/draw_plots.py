#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# draw_plots.py

Module (core): Functions for drawing graphs.

"""
from __future__ import annotations
from typing import Optional, Iterable

import pandas as pd
import numpy as np
import logging

from scipy import stats
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib.font_manager as fm
from IPython.display import HTML
import webbrowser
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    normalize)
from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter


from utils import funcs as func

def set_pub():
    rcParams.update({
        "font.weight": "bold", 
        "axes.labelweight": 'bold',
        'figure.dpi': '300',
        'axes.facecolor': 'w',
        "axes.labelsize": 15,   
        "lines.linewidth": 1,   
        "lines.color": "k",     
        "grid.color": "0.5",    
        "grid.linestyle": "-",  
        "grid.linewidth": 1.0,  
        "savefig.dpi": 300,     
    })


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

class Plot(object):
    def __init__(self, 
                 data: pd.DataFrame, 
                 legend: Optional[dict]={},
                 colors: Optional[Iterable]=None,
                 events: Optional[Iterable]=None,
                 dpi = 300,
                 **kwargs
                 ):
        
        assert isinstance(data, pd.DataFrame)
        
        self.cmap = plt.get_cmap('viridis')
        self.data = data
        
        
        self.legend = legend
        self.colors = colors
        self.events = events
        self.dpi = 500
        self.save_dir = '/Users/flynnoconnell/Pictures/plots'    
        self.facecolor = 'white'
        self.checks = {}


    def PCA(self,
            ret: bool = False,
            noscale: bool=True,
            title: str = '',
            caption: str = '',
            norm=False,
            ss: bool = False,
            mm: bool = False,
            rs: bool = False,
            numcomp: Optional[int] = 2, 
            size=5,
            facecolor: str = 'white',
            ns_save: bool = False,
            ss_save: bool = False,
            rs_save: bool = False,
            mm_save: bool = False,
            remove_outliers: bool=False,
            std: int=3,
            oval: bool = False,
            skree: bool = False
            ):
        
        
        data = self.data.copy()
        colors = self.colors.copy()

        
        if remove_outliers:
            
            df2 = data.copy()
            colors2 = colors.copy()
            
            df2.reset_index(drop=True,inplace=True)
            colors2.reset_index(drop=True, inplace=True)
            ind = (np.abs(stats.zscore(df2)) < 2).all(axis=1)
            df2 = df2[ind]
            colors2 = colors2[ind]
            
            data = df2
            self.colors = colors2
        pca = PCA(n_components=numcomp)
        
        if norm:
            data = normalize(data, axis=0)
            self.checks['norm'] = data
            
        if noscale: 
            
            if ns_save: 
                savefig=True
                msg='no scale '
            else:
                savefig=False
                msg=''
            colors = self.colors
            ns_data = pca.fit_transform(data)
            
            variance = np.round(
                pca.explained_variance_ratio_ * 100, decimals=1)
            
            labels = [
                'PC' + str(x) + f' - {variance[x-1]}%' for x in range(1, len(variance) + 1)]
            
            df = pd.DataFrame(ns_data, columns=labels)
            # if remove_outliers:
            #     df.reset_index(drop=True,inplace=True)
            #     df2 = df.copy()
            #     df2['colors'] = colors.copy()
            #     df = df2[(np.abs(stats.zscore(df)) < std).all(axis=1)]
            #     colors = df.pop('colors')

            self.scatter(
                  plot_df=df,
                  color_dict=self.legend,
                  c_points=colors,
                  title=title,
                  caption=caption, 
                  size=size,
                  savefig=savefig,
                  msg=msg )
            if skree: 
                self.plot_skree(variance, title=title)
            logging.info('No scale')
        
        if ss: 
            if ss_save: 
                savefig=True
                msg='ss-scale'
            else:
                savefig=False
                msg=''

            colors = self.colors
            ss_data = StandardScaler().fit_transform(data)
            ss_pca = pca.fit_transform(ss_data)
                        
            variance = np.round(
                pca.explained_variance_ratio_ * 100, decimals=1)
            
            labels = [
                'PC' + str(x) + f' - {variance[x-1]}%' for x in range(1, len(variance) + 1)]
            
            df = pd.DataFrame(ss_pca, columns=labels)
            if remove_outliers:
                df.reset_index(drop=True,inplace=True)
                df2 = df.copy()
                df2['colors'] = colors.copy()
                df = df2[(np.abs(stats.zscore(df)) < std).all(axis=1)]
                colors = df.pop('colors')

            self.scatter(
                  plot_df=df,
                  color_dict=self.legend,
                  c_points=colors,
                  title=title,
                  caption=caption, 
                  size=size,
                  savefig=savefig,
                  msg=msg
                  )
            logging.info('ss')
            
        if rs: 
            if rs_save: 
                savefig=True
                msg='rs-scale'
            else:
                savefig=False
                msg=''


            colors = self.colors
            rs_data = RobustScaler().fit_transform(data)
            rs_pca = pca.fit_transform(rs_data)
            
            variance = np.round(
                pca.explained_variance_ratio_ * 100, decimals=1)
            labels = [
                'PC' + str(x) + f' - {variance[x-1]}%' for x in range(1, len(variance) + 1)]
            df = pd.DataFrame(rs_pca, columns=labels)
            if remove_outliers:
                df.reset_index(drop=True,inplace=True)
                df2 = df.copy()
                df2['colors'] = colors.copy()
                df = df2[(np.abs(stats.zscore(df)) < std).all(axis=1)]
                colors = df.pop('colors')
                
            self.scatter(
                  plot_df=df,
                  color_dict=self.legend,
                  c_points=colors,
                  title=title,
                  caption=caption, 
                  size=size,
                  savefig=savefig,
                  msg=msg
                  )
            logging.info('rs')
        if mm: 
            if mm_save: 
                savefig=True
                msg='mm-scale '
            else:
                savefig=False
                msg=''


            colors = self.colors
            mm_data = MinMaxScaler().fit_transform(data)
            mm_pca = pca.fit_transform(mm_data)
            
            variance = np.round(
                pca.explained_variance_ratio_ * 100, decimals=1)
            labels = [
                'PC' + str(x) + f' - {variance[x-1]}%' for x in range(1, len(variance) + 1)]
            
            df = pd.DataFrame(mm_pca, columns=labels)
            if remove_outliers:
                df.reset_index(drop=True,inplace=True)
                df2 = df.copy()
                df2['colors'] = colors.copy()
                df = df2[(np.abs(stats.zscore(df)) < std).all(axis=1)]
                colors = df.pop('colors')
                
            self.scatter(
                  plot_df=df,
                  color_dict=self.legend,
                  c_points=colors,
                  title=title,
                  caption=caption, 
                  size=size,
                  savefig=savefig,
                  msg=msg
                  )
            logging.info('mm')
        if ret: 
            return df, labels, variance
        else:
            
            return None

    def scatter(self,
                plot_df: pd.DataFrame=None,
                color_dict: dict=None,
                c_points: pd.Series=None,
                title: Optional[str] = None,
                plottype: Optional[str]='Scatter',
                oval: bool= False,
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
                df_colors (pd.Series) | (np.1darray): Colors for each scatter point.
                title (str): Text to display as title.
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
        
        if isinstance(plot_df, pd.DataFrame):
            df = plot_df
        else:
            df = self.data.copy()
            
        if color_dict: 
            color_dict = color_dict
        else:
            color_dict = self.legend
            
        if c_points is not None: 
            c_points = c_points
        
        else:
            c_points = self.colors
            
        if title is None:
            title = 'Scatter Plot'
            
        if 'colors' in df.columns:
            c_points = df.pop('colors')
                            
        fig = plt.figure()
        fig.set_dpi(300)
        
        x = df.iloc[:,0]
        y = df.iloc[:,1]

        ax = fig.gca()
        
        ax.set_xlabel(df.columns[0], weight='bold')
        ax.set_ylabel(df.columns[1], weight='bold')
        ax.patch.set_facecolor = 'white'
        
        ax.scatter(x, y,
                    c=c_points,
                    s=size,
                    marker=marker,
                    alpha=alpha,
                    facecolor='white'
                    )
        
        
        df.reset_index(drop=True, inplace=True)
        c_points.reset_index(drop=True, inplace=True)
        
        df['colors'] = c_points
        if oval:
            for color in np.unique(c_points):
                
                _df = df.loc[(df['colors'] == color)]
                
                func.confidence_ellipse(
                    _df.iloc[:,0],
                    _df.iloc[:,1],
                    ax,
                    facecolor=color,
                    edgecolor='k',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.08)

        ax.set_title(f'{title}', fontweight = 'bold')

        proxy, label = func.get_handles(color_dict, marker = 'o', markersize=5, alpha=1)
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
            #init
        return None
            
    def plot_skree(self,
                    var: np.ndarray,
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

    def scatter_2d(x,
                      colors,
                      color_dict,
                      y=None,
                      z=None,
                      label_x=None,
                      label_y=None,
                      label_z=None,
                      title: str = ''):
        """ 
        Make scatter plot, given data and labels.  
        
        """
        
        if isinstance(x, pd.DataFrame):
            
            x = x.iloc[:,0]
            y = x.iloc[:,1]
            
            label_x = x.columns[0]
            label_y = x.columns[1]
        else: 
            x=x
            y=y
            
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        ax.set_xlabel(label_x, weight='bold') 
        ax.set_ylabel(label_y, weight='bold') 
        ax.set_zlabel(label_z, weight='bold') 
        ax.set_title('{title}')
        ax.scatter(x, y, z, s=40, c=colors, marker='o', alpha=1)
        
        proxy, labels = func.get_handles(color_dict, linestyle='none')
                
        ax.legend(handles=proxy,
                  labels=labels, 
                  loc='upper right',
                  prop={'size':6},
                  bbox_to_anchor=(1,1), 
                  ncol=2, numpoints=1)
        
        return fig
    
    def scatter_3d(self,
                plot_df: pd.DataFrame=None,
                color_dict: dict=None,
                c_points: pd.Series=None,
                title: Optional[str] = None,
                plottype: Optional[str]='Scatter',
                oval: bool= False,
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
                df_colors (pd.Series) | (np.1darray): Colors for each scatter point.
                title (str): Text to display as title.
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
        
        if isinstance(plot_df, pd.DataFrame):
            df = plot_df
        else:
            df = self.data.copy()
            
        if color_dict: 
            color_dict = color_dict
        else:
            color_dict = self.legend
            
        if c_points is not None: 
            c_points = c_points
        
        else:
            c_points = self.colors
            
        if title is None:
            title = 'Scatter Plot'
            
        if 'colors' in df.columns:
            c_points = df.pop('colors')
                            
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
                    c=c_points,
                    s=size,
                    marker=marker,
                    alpha=alpha,
                    facecolor='white'
                    )
        
        
        df.reset_index(drop=True, inplace=True)
        c_points.reset_index(drop=True, inplace=True)
        
        df['colors'] = c_points
        if oval:
            for color in np.unique(c_points):
                
                _df = df.loc[(df['colors'] == color)]
                
                func.confidence_ellipse(
                    _df.iloc[:,0],
                    _df.iloc[:,1],
                    ax,
                    facecolor=color,
                    edgecolor='k',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.08)

        ax.set_title(f'{title}', fontweight = 'bold')

        proxy, label = func.get_handles(color_dict, marker = 'o', markersize=5, alpha=1)
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
        x = df.iloc[:,0]
        y = df.iloc[:,1]
        z = df.iloc[:,2]
        
        fig = plt.figure()

        ax = fig.gca(projection='3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('{}'.format(session) + ' ' + 'PCA')
        ax.scatter(x, y, z, c=colors, s=size, marker=marker, alpha=alpha)

        proxy, label = func.get_handles(color_dict)
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

    def plot_session(cells,
                      signals: pd.DataFrame,
                      time: pd.Series,
                      session: str,
                      numlicks: int,
                      timestamps: dict = None,
                      lickshade: int = 1,
                      save_dir: bool = True) -> None:
        
        # create a series of plots with a shared x-axis
        fig, axs = plt.subplots(len(cells), 1, sharex=True, facecolor='white')
        for i in range(len(cells)):
            # get calcium trace (y axis data)
            signal = list(signals.iloc[:, i])

            # plot signal
            axs[i].plot(time, signal, 'k', linewidth=.8)

            # Get rid of borders
            axs[i].get_xaxis().set_visible(False)
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].set_yticks([])  # no Y ticks

            # add the cell name as a label for this graph's y-axis
            axs[i].set_ylabel(signals.columns[i],
                              rotation='horizontal', labelpad=15, y=.1, fontweight='bold')

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
        # plt.legend(loc=(1.04, 1))
        axs[-1].get_xaxis().set_visible(True)
        axs[-1].spines["bottom"].set_visible(True)
        # fig.text(0, -.03, 'Number of Licks: {}'.format(numlicks),
        #          fontstyle='italic', fontsize='small')
        if save_dir:
            fig.savefig(f'/Users/flynnoconnell/Pictures/plots/{session}_zm.png',
                        bbox_inches='tight', dpi=600, facecolor='white')
            logger.info(msg='Session figure saved.')

        return None

    def plot_zoom(nplot,
                  tracedata,
                  time,
                  timestamps,
                  session,
                  colors,
                  zoomshade: float = 0.2,
                  save_dir: Optional[bool] = True
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
            fig.savefig(f'/Users/flynnoconnell/Pictures/plots/{session}_zm.png',
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
                  my_stim: list = None,
                  save_dir: str = None
                  ) -> None:
        
        if my_stim:
            for stim in my_stim:
                trial_times = { stim: trial_times[stim] for stim in my_stim }
            
        for stim, times in trial_times.items():
            trialno = 0
            for trial in times:
                trialno += 1

                # get only the data within the analysis window
                data_ind = np.where((time > trial - 1) & (time < trial + 3))[0]
                # index to analysis data
                this_time = time[data_ind]

                fig, ax = plt.subplots(nplot, 1, sharex=True)
                for i in range(nplot):
                    # Get calcium trace for this analysis window
                    signal = list(tracedata.iloc[data_ind, i])
                    # plot signal
                    ax[i].plot(this_time, signal, 'k', linewidth=1)
                    ax[i].get_xaxis().set_visible(False)
                    ax[i].spines["top"].set_visible(False)
                    ax[i].spines["bottom"].set_visible(False)
                    ax[i].spines["right"].set_visible(False)
                    ax[i].set_yticks([])
                    ax[i].set_ylabel(tracedata.columns[i],
                                      rotation='horizontal',
                                      labelpad=15, y=.1)
                    # Add shading.
                    for stimmy in ['Lick', 'Rinse', stim]:
                        done = 0
                        timey = timestamps[stimmy]
                        for ts in timey:
                            if trial - 1 < ts < trial + 3:
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

