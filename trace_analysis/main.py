# -*- coding: utf-8 -*-

"""
#main.py

Module: Code execution.
"""
import logging
from core.calciumdata import CalciumData

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# %% Initialize data

datadir = '/Users/flynnoconnell/Documents/Work/Data'
animal2 = 'PGT08'
date02 = '070121'
date03 = '071621'
date04 = '072721'

animal = 'PGT13'
date = '011222'
date01 = '011422'
date001 = '121021'

data1 = CalciumData(animal, date, datadir)
data11 = CalciumData(animal, date01, datadir)
data111 = CalciumData(animal, date001, datadir)

data2 = CalciumData(animal2, date02, datadir)
data22 = CalciumData(animal2, date03, datadir)
data222 = CalciumData(animal2, date04, datadir)


#%%
alldata = CalciumData.alldata
print(alldata)

dct1 = {
        'PGT13': {'Date1': data1,
                  'Date2': data2},
        'PGT08': {'Date1': data1,
                  'Date2': data2}
        
        }
print("\n".join(f"{key} - {len(value)} sessions." for key, value in dct1.items()))
print(dct1.items())
print(alldata.items())


# len(alldata)


if __name__ == "__main__":
    pass     








