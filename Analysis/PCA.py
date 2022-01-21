#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:28:23 2022

@author: flynnoconnell
"""
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as ss


@dataclass 
class PlotPCA:
    allcells: list
    colors_fill: list
    trc: pd.DataFrame
    
    def getcolors(colors_fill, allcells):
        pca_colors = []
        i = 0 
        for i in range(0, len(allcells)):
            if len(allcells) > len(colors_fill):
                print("You need to add more colors, too many cells not enough colors")
                sys.exit()
            else:
                i += 1
                pca_colors.append(colors_fill[i])
        return pca_colors
    
    def pcaCells(trc, cell_names):
        scaled_trc = ss().fit_transform(trc.T)
        pca = PCA()
        pca.fit(scaled_trc) # calc loading scores and variation 
        pca_trc = pca.transform(scaled_trc) # final transform
        per_var = np.round(pca.explained_variance_ratio_*100, decimals=1) 
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
        pca_df = pd.DataFrame(pca_trc, index=cell_names, columns=labels)
        
        return pca_df, per_var, labels

    def pcaTime(trc, time, labels):
        scaled_trc = ss().fit_transform(trc)
        pca = PCA()
        pca.fit(scaled_trc) # calc loading scores and variation 
        pcaT_trc = pca.transform(scaled_trc) # final transform
        perT_var = np.round(pca.explained_variance_ratio_*100, decimals=1) 
        labelsT = ['PC' + str(x) for x in range(1, len(perT_var)+1)]
        pcaT_df = pd.DataFrame(pcaT_trc, index=time, columns=labels)

        return pcaT_df, perT_var, labelsT

        
