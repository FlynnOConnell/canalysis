# -*- coding: utf-8 -*-
"""
# process_gpio.py

"""
import os

import numpy as np
import pandas as pd


class GpioData


# %% User paramters

# input must be a .csv file of gpio events exported from inscopix data processing software
gpio_file = r'A:\PGT08\032422\PGT08_032422_gpio.csv'

# Dict used to decode stimuli. key is stimulus name and value is which inputs are active for the given stimulus.
# DAQ takes inputes 1-4
decode_gpio = {
    'Rinse': [1, 2, 3, 4],
    'ArtSal': [1, 3, 4],
    'Quinine': [1, 2, 4],
    'NaCl': [1, 3],
    'Sucrose': [1, 2],
    'MSG': [1, 4],
    'Citric': [1, 2, 3],
}

# Threshold for detecting an ON TTL signal, if error, raise this value
threshold = 3000

# distance timestamps can be adjusted to the lick
ts_adjust = .005

# %% initilizations

timestamps = {}

# Function to get all timestamps out, from array arr, which fall between ts+adjust and ts-adjust

def within(arr, ts, adjust):
    arr = np.array(arr)
    out = arr[np.where((arr > ts-adjust) & (arr < ts+adjust))]
    return out


# Fetch data
data = pd.read_csv(gpio_file)

# get end of recording
rec_end = max(data.loc[np.where(
    data[' Channel Name'] == ' BNC Sync Output')[0], 'Time (s)'])

# trim out all data that is not GPIO
data = data.iloc[np.where(
    data[' Channel Name'].str.contains('GPIO'))[0], :]
outpath = os.path.join(os.path.splitext(gpio_file)[0]+'_processed.csv')

# %% Obtaining GPIO Timestamps

for chan in pd.unique(data[' Channel Name']):
    print('Processing events for:', chan)
    event = []  # container for timestamps
    # get only data for this GPIO
    gp_chan = data.iloc[np.where(data[' Channel Name'] == chan)[0], :]
    # get all values which pass the threshold
    gp_check = gp_chan[gp_chan[' Value'] > threshold]
    for index, row in gp_check.iterrows():  # for each value
        t = row['Time (s)']  # get the time
        # tcomp checks to see if any times in the last 11 ms exceed the threshold.
        # Since pulses last 10 ms, this should not return any true values if it is the pulse start
        tcomp = gp_chan[(gp_chan['Time (s)'] >= t-.011) &
                        (gp_chan['Time (s)'] < t)][' Value'] > threshold
        # if all checks are good add the event
        if row[' Value'] > threshold and not np.any(tcomp):
            event.append(t)
    timestamps[chan] = event
timestamps['Lick'] = timestamps[' GPIO-1']
# need to add a portion here to confirm that all licks occur withn 100 ms or MORE and fix timestamps that are slightly off

# %%running checks and data cleaning, including syncing timestamps

for i in range(2, 5):  # for channels 2 through 4
    for ts in timestamps[' GPIO-{}'.format(i)]:
        # if the timestamp is not in channel 1 (all licks)
        if ts not in timestamps[' GPIO-1']:
            # get all nearby timestamps from GPIO1
            tscheck = within(timestamps[' GPIO-1'], ts, ts_adjust)
            if len(tscheck) == 1:  # there should only be one
                # if so replace the timestamp with the one from GPIO-1
                print(
                    'Adjusted GPIO-{} timestamps from {} to {}'.format(i, ts, tscheck[0]))
                timestamps[' GPIO-{}'.format(i)].remove(ts)
                timestamps[' GPIO-{}'.format(i)].append(tscheck[0])
            # else:
            #     sys.exit(
            #         'Timestamp {} has no nearby timestamp in GPIO1 but is in GPIO{}'.format(ts, i))
    timestamps[' GPIO-{}'.format(i)].sort()  # sort the timestamps

allts = []
# this loop collects all GPIO timestamps
for chan in pd.unique(data[' Channel Name']):
    allts.extend(timestamps[chan])


# %% Decoding GPIO into stimuli

# loop through the stimulus decoder dictionary
for stim, inputs in decode_gpio.items():
    ts_holder = []
    for ts in timestamps[' GPIO-1']:  # for each lick
        for chan in range(1, 5):
            if ts not in timestamps[' GPIO-{}'.format(chan)] and chan in inputs:
                break  # if the input is needed to code for this stimulus and isnt present, move on
            elif ts in timestamps[' GPIO-{}'.format(chan)] and chan not in inputs:
                break  # if the input is not needed to code for this stimulus and is present, move on
        else:
            # if the timestamp belongs in this stimulus, save it
            ts_holder.append(ts)
    timestamps[stim] = ts_holder  # add the timestamps to the list

# %% Formatting output

alldata = []
for ts in timestamps['Lick']:
    # generate time for event and for a 0 signal 10 ms later (on pulse lasts 10 ms)
    this_dict = {'Time(s)': ts}
    follow_dict = {'Time(s)': round(ts+.01, 3)}
    # this loop goes through each stimulus and adds a 1 if it is active
    for stim in ['Lick']+list(decode_gpio.keys()):
        if ts in timestamps[stim]:
            this_dict[stim] = 1
        else:
            this_dict[stim] = 0
        follow_dict[stim] = 0  # all signals should be 0 after pulse
    # add the data
    alldata.append(this_dict)
    alldata.append(follow_dict)
# convert to dataframe
output = pd.DataFrame(alldata)
# add 2 empty rows for the end and beginning of recording
output.loc[len(output)] = 0
output.loc[len(output)] = 0
# set the second one to the recording end time
output.loc[len(output)-1, 'Time(s)'] = rec_end
# sort
output.sort_values('Time(s)', inplace=True)
# output to a csv file to be imported into inscopix data processing
output.to_csv(outpath, index=False)
follow_dict[stim] = 0  # all signals should be 0 after pulse
# add the data
alldata.append(this_dict)
alldata.append(follow_dict)
# convert to dataframe
output = pd.DataFrame(alldata)
# add 2 empty rows for the end and beginning of recording
output.loc[len(output)] = 0
output.loc[len(output)] = 0
# set the second one to the recording end time
output.loc[len(output)-1, 'Time(s)'] = rec_end
# sort
output.sort_values('Time(s)', inplace=True)
# output to a csv file to be imported into inscopix data processing
output.to_csv(outpath, index=False)
print('File done processing.')
