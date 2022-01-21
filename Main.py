# -*- coding: utf-8 -*-

"""
@author: Dr. Patricia Di Lorenzo Laboratory @ Binghamton University
"""
from matplotlib.collections import LineCollection
import Analysis.Func as ca
# import Plots as Plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import easygui
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

print('__name__ is {0}'.format(__name__))
if __name__ == "__main__":
    print("Hello World!")
# %% User parameters

# Set user OS
Mac = 0

doStats = 1        # Perform / output statistics.
outputStats = 0    # Output stats to excel file.
doPCA = 0          # Do PCA Analysis
wholeSession = 0   # Plot "lick" graph for whole session.
zoomGraph = 0      # Graph to zoom in on time window.
stimGraph = 0      # All cells around single stimulus.
singleCells = 0    # Individual cells around a single stimulus.
showPlots = 0      # Show graphs in plots tab as they are drawn.
savePlots = 0      # Save plots to parent dir.
ani = 0

animalID = 'PGT08'
session_date = '070121'


# How much area after each lick (whole session, (s)).
lickshade = 1
# How much area after each lick (zoomed, (s)).
zoomshade = .2

# %% Initialization
# Defining the colors for each tastant, used on graphs and legends
colors = {
    'ArtSal': 'blue',
    'Citric': 'yellow',
    'Lick': 'darkgray',
    'MSG': 'orange',
    'NaCl': 'green',
    'Quinine': 'red',
    'Rinse': 'lightsteelblue',
    'Sucrose': 'purple'
}

colors_fill = ['b', 'g', 'r', 'c', 'm', 'y',
          'k', 'lime', 'darkorange', 'purple', 'magenta', 'crimson', 'cyan', 'peru']

session = (animalID + '_' + session_date)
data_dir: str
# Directory management. 

parent_dir = r'R:\Flynn\Calcium_Imaging\Results'
data_dir = 'A:\\'
test_dir = r"C:\Users\dilorenzo\Desktop\Calcium Plots"

if Mac == 1:
    parent_dir = os.path.expanduser("~/Documents/Work/Results")
    data_dir = os.path.expanduser("~/Documents/Work/Data")

traces = os.path.join(data_dir,
                      session,
                      session
                      + '_ROI_traces.csv')
events = os.path.join(data_dir,
                      session,
                      session
                      + '_gpio_processed.csv')

# Nested directories for saving plots.
animal_dir = os.path.join(parent_dir, animalID)
session_dir = os.path.join(animal_dir, session_date)
stats_dir = os.path.join(session_dir, 'Stats')

cell_dir = os.path.join(session_dir, 'Individual Cells')
tastant_dir = os.path.join(session_dir, 'Tastant Trials')

# Create parent/session path if not already a directory.
if not os.path.isdir(parent_dir):
    os.mkdir(parent_dir)
if not os.path.isdir(animal_dir):
    os.mkdir(animal_dir)
if not os.path.isdir(session_dir):
    os.mkdir(session_dir)

# Populate main DataFrames.
tracedata = pd.read_csv(traces, low_memory=False)
eventdata = pd.read_csv(events)

# Initialize lists.
allstats, lickstats, statsummary = [], [], []

tracedata, tracedata_out, trc = ca.Traces.PopData()
assert isinstance(eventdata, object)
timestamps, allstim, drylicks, licktime, trial_times = ca.Events.PopEvents(eventdata)
time = np.array(tracedata.iloc[:, 0])

lickevent = []
for lick in licktime: 
    ind = ca.Traces.getMatchedTime(time, lick)
    match = time[ind]
    lickevent.append(match[0])

# List of cells for chosing single-cells
cells = tracedata.columns[1:]
allcells = [x.replace(' ', '') for i, x in enumerate(cells)]

# Setting cell for individual graphs
if singleCells == 1:
    plotCells = []
    while True:
        myPlotCells = easygui.multchoicebox(
            msg='Please select cell(s) to plot, or cancel to continue.',
            choices=allcells)
        if myPlotCells is None:  # If no cells chosen
            ynBox = easygui.ynbox(
                msg=('No cells chosen, continue without'
                     'plotting individual cells?'),
                choices=(
                    'Yes', 'No, take me back'))
            if ynBox is False:
                continue
            if ynBox is True:
                singleCells = 0
                break
        else:
            plotCells.append(myPlotCells)
            break

# Make dictionary, so we can pull an integer to index the cell later
cell_index = {x: 0 for x in allcells}
cell_index.update((key, value) for value, key in enumerate(cell_index))
cell_names = tracedata.columns.values.tolist()
cell_names.pop(0)

# Fetch some numbers (cells, time np.array, number of licks)
nplot = tracedata.shape[1]-1
binsize = time[2]-time[1]
numlicks = len(timestamps['Lick'])
print('Number of Licks in this session:', numlicks)

# %% Functions
def uniquify(path):
    """ Not yet used. Make unique filename if path already exists. 
    """
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path

# %% Lick analysis
lickStats = pd.DataFrame(columns=[
    'File', 'Cell', 'Lick type'])

# Pull lick data methods
start_traces, end_traces, bouts_td_time, bout_dff = ca.Lick.getBout(licktime, tracedata, time, nplot)
antibout_dff = ca.Lick.getAntiBout(tracedata, bouts_td_time, time)
sponts, stdevs = ca.Lick.getSpont(licktime, tracedata, time)

idxs = np.where(np.diff(licktime) > 30)[0]
spont_intervs = np.column_stack((licktime[idxs], licktime[idxs+1]))
sponts, stdevs = [], []
for cell in tracedata.columns[1:]:
    print(cell)
    cell_spont = []
    cell_sd = []
    for bound_low, bound_high in spont_intervs:
        ind0 = ca.Traces.getMatchedTime(time, bound_low)
        ind1 = ca.Traces.getMatchedTime(time, bound_high)
        cell_spont.append(
            np.mean(np.array(tracedata.loc[ind0[0]:ind1[0], cell])))
        cell_sd.append(
            np.std(np.array(tracedata.loc[ind0[0]:ind1[0], cell])))
    mean_spont = np.mean(cell_spont)
    sponts.append(mean_spont)
    mean_stdev = np.mean(cell_sd)
    stdevs.append(mean_stdev)

# Calculate stats
cell_id = np.array(tracedata.columns[1:])
for index, cell in enumerate(cell_id):
    print(index)
    if (bout_dff[index]) > ((sponts[index])+((stdevs[index])*2.58)):
        licktype = 'BOUT'
    elif (bout_dff[index]) < ((sponts[index])-((stdevs[index])*2.58)):
        licktype = 'ANTI-LICK'
    else:
        licktype = 'non-lick'
    lick_dict = {
        'File': session,
        'Cell': cell,
        'Lick cell': licktype}
    lickStats = lickStats.append(
        lick_dict, ignore_index=True)
lickstats.append(lickStats)
                     


# %% Generate graph for entire session
if wholeSession == 1:
    fig = ca.Plots.lineSession(nplot, tracedata, time, timestamps, lickshade, session, numlicks)
    if savePlots == 1:
        fig.savefig(session_dir+'/{}_session.png'.format(session),
                    bbox_inches='tight', dpi=300)
    if showPlots == 1:
        plt.show()
    else:
        plt.close()

# %% Generate graph zoomed to a specific area
if zoomGraph == 1:
    zoomfig = ca.Graph.lineZoom(nplot, tracedata, time, timestamps, session, zoomshade, colors)
    if savePlots == 1:
        zoomfig.savefig(
            session_dir+'/{}_zoomed.png'.format(session),
            bbox_inches='tight',
            dpi=300)
    if showPlots == 1:
        plt.show()
    else:
        plt.close()

# %% Generate graph centered around each stimulus delivery
if stimGraph == 1:
    fig, stim, trialno = ca.Graph.lineStim(nplot, tracedata, time, timestamps, trial_times, session, colors)
    if savePlots == 1:
        if not os.path.isdir(tastant_dir):
            os.mkdir(tastant_dir)
        fig.savefig(tastant_dir+'/{}_{}_trial{}.png'.format(
            session, stim, trialno),
            bbox_inches='tight',
            dpi=300)
    if showPlots == 1:
        plt.show()
    else:
        plt.close()

# %% Generate graph of single cells
if singleCells == 1:
    for my_cell in myPlotCells:
        fig, my_cell = ca.Graph.lineSingle(myPlotCells, cell_index, tracedata, time, timestamps, trial_times, session, colors)

        if savePlots == 1:
            # Create a directory for each cell, if there isn't one already
            cellPath = os.path.join(cell_dir, my_cell)
    
            if not os.path.isdir(cell_dir):
                os.mkdir(cell_dir)
            if not os.path.isdir(cellPath):
                os.mkdir(cellPath)
    
            fig.savefig(cellPath+'/{}_{}_{}.png'.format(
                my_cell, session, stim),
                bbox_inches='tight',
                dpi=300)
    
        if showPlots == 1:
            plt.show()
        else:
            plt.close()

# %% Generate statistics
if doStats == 1:
    trialStats, summary = ca.Traces.Stats(tracedata, trial_times, time, tracedata_out, session)
    allstats.append(trialStats)
    statsummary.append(summary)

# %% Data Post-Processing
if outputStats == 1:
    ca.output(session, session_date, stats_dir, allstats, statsummary, lickstats, tracedata_out)
    
# %% PCA Analysis
# Transform Data
if doPCA == 1:
    pca_df, per_var, labels = ca.Graph.pcaCells(trc, cell_names)
    pca_dfT, perT_var, labelsT = ca.Graph.pcaTime(trc, time, labels)
        
    pca_df['colors'] = ca.Graph.getColors(colors_fill, allcells)
    pca_df['Bout'] = bout_dff
    pca_df['Antibout'] = antibout_dff
    
    licknolick = []
    for t in time: 
        tmp = []
        for bout in bouts_td_time: 
            if t > bout[0] and t < bout[1]: 
                licknolick.append(1)
                tmp.append(t)
            else:
                continue
        if not tmp: 
            licknolick.append(0)
        
    pca_dfT['Licks'] = licknolick         
            
    # Bar Skree------------------
    plt.figure()
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Variance explained (%)')
    plt.xlabel('Principal Component')
    plt.title('{}'.format(session) + '-' + 'Scree plot (Bar)')
    plt.xticks(range(1,len(per_var)+1))
    plt.show()
    
    # Line Skree------------------
    plt.figure()
    plt.plot(labels, per_var, 'o-', linewidth=2, color='blue')
    plt.title('{}'.format(session) + '-' + 'Scree plot (Line)')
    plt.ylabel('Variance explained (%)')
    plt.xlabel('Principal Component')
    plt.xticks(range(1,len(per_var)+1))
    plt.show()
    
    # 2d scatter------------------
    plt.figure()
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('{}'.format(session) + '-' + 'PCA 1 and 2')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    plt.show()
    
    # 3d scatters--------------------
    fig = ca.Graph.Scatter(
        pca_df.PC1,  pca_df.PC2, pca_df.PC3,
        'PC1 - {0}%'.format(per_var[0]),
        'PC2 - {0}%'.format(per_var[1]),
        'PC3 - {0}%'.format(per_var[2]))
    
    fig = ca.Graph.Scatter(
        pca_df.PC1, pca_df.Bout, pca_df.Antibout,
        'PC1 - {0}%'.format(per_var[0]),
        'Avg. dF/F (Bout)',
        'Avg. dF/F (Antibout)')
    
    if ani ==1:
        fig = ca.Graph.aniScatter(
            pca_df.PC1, pca_df.Bout, pca_df.Antibout,
            'PC1 - {0}%'.format(per_var[0]),
            'Avg. dF/F (Bout)',
            'Avg. dF/F (Antibout)')
    
    pca_dfT['Time'] = time
    pca_dfT.reset_index(drop=True, inplace=True)
    x = pca_dfT['PC1'].iloc[0:1202]
    y = pca_dfT['PC2'].iloc[0:1202]
    z = pca_dfT['PC3'].iloc[0:1202]
    
    
    
    
    
    lines = [((x0,y0), (x1,y1))
             for x0, y0, x1, y1
             in zip(x[:-1], y[:-1], x[1:], y[1:])]
    
    fig = ca.Graph.tsScatter(
        x,  y, z,
        'PC1 - {0}%'.format(per_var[0]),
        'PC2 - {0}%'.format(per_var[1]),
        'PC3 - {0}%'.format(per_var[2]))
    
    fig = ca.Graph.aniScatter(
        x,  y, z,
        'PC1 - {0}%'.format(per_var[0]),
        'PC2 - {0}%'.format(per_var[1]),
        'PC3 - {0}%'.format(per_var[2]))
    
    
    # set up colors 
    c = ['r' if a > 0 else 'k' for a in licknolick]
    
    # convert time series to line segments
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
    colored_lines = LineCollection(lines, colors=c, linewidths=(.5,))
    # plt.legend(handles=handles, labels=list(pca_df.index.values), 
    #        loc='best', prop={'size':6}, bbox_to_anchor=(1,1),ncol=1, numpoints=1)
    
    fig, ax = plt.subplots(1)
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    plt.show()

