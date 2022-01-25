# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:55:34 2022

@author: flynnoconnell
"""

import os
import pandas as pd
import numpy as np
import math


def dup_check(signal, peak_signal):
    checker = []
    for value in signal:
        dupcheck = math.isclose(value, peak_signal, abs_tol=1.0)
        if dupcheck is True:
            checker.append(dupcheck)
    if not checker:
        raise Exception(
            'A duplicate time may have been chosen for time of peak')


def uniquify(path):
    """ Not yet used. Make unique filename if path already exists.
    """
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path


def clean(df):
    accepted = np.where(df.loc[0, :] == ' accepted')[0]
    df_clean = df.iloc[:, np.insert(accepted, 0, 0)]
    df_clean = df_clean.drop(0)
    df_clean = df_clean.astype(float)
    df_clean = df_clean.rename(columns={' ': 'Time(s)'})
    df_clean = df_clean.reset_index(drop=True)
    df_clean.columns = [column.replace(' ', '') for column in df_clean.columns]

    return df_clean


def get_peak_window(time, peak_time):
    """ Returns the index of tracedata centered 1s around the peak flourescent value for
        that trial.
    """
    time: list
    peak_time: float

    aux, window_ind = [], []
    for valor in time:
        aux.append(abs(peak_time - valor))

    window_ind.append(aux.index(min(aux)) - 20)
    window_ind.append(aux.index(min(aux)) + 20)

    return window_ind


def get_matched_time(time, *argv):
    """  Finds the closest number in tracedata time to the input.
    """
    time: list

    return_index = []
    for arg in argv:
        temp = []
        for valor in time:
            temp.append(abs(arg - valor))
        return_index.append(temp.index(min(temp)))
    return return_index


def pop_events(eventdata):
    # Get timestamps
    timestamps = {}
    allstim = []
    for stimulus in eventdata.columns[1:]:
        timestamps[stimulus] = list(
            eventdata['Time(s)'].iloc[np.where(
                eventdata[stimulus] == 1)[0]])
        if stimulus != 'Lick':
            allstim.extend(timestamps[stimulus])

    # Get drylicks
    drylicks = [x for x in timestamps['Lick'] if x not in allstim]
    licktime = eventdata['Time(s)'].iloc[
        np.where(eventdata['Lick'] == 1)[0]]
    licktime = licktime.reset_index(drop=True)

    # Populate trials
    trial_times = {}
    for stim, tslist in timestamps.items():
        if stim != 'Lick' and stim != 'Rinse' and len(timestamps[stim]) > 0:
            # Get list of tastant deliveries
            tslist.sort()
            tslist = np.array(tslist)
            times = [tslist[0]]  # First one is a trial
            # Delete duplicates
            for i in range(0, len(tslist) - 1):
                if tslist[i] == tslist[i + 1]:
                    nxt = tslist[i + 1]
                    tslist = np.delete(tslist, i + 1)
                    print(f"Deleted timestamp {nxt}")
            # Add each tastant delivery preceded by a drylick (trial start).
            for ts in tslist:
                last_stimtime = tslist[np.where(
                    tslist == ts)[0] - 1]
                last_drytime = drylicks[np.where(
                    drylicks < ts)[0][-1]]
                if last_drytime > last_stimtime:
                    times.append(ts)
            trial_times[stim] = times

    return timestamps, allstim, drylicks, licktime, trial_times


def get_bout(licktime, tracedata, time, nplot):
    # Make bout intervals ----------------
    tmp, bout_intervs, boutstart, boutend, start_traces, end_traces = [], [], [], [], [], []
    for i, v in enumerate(licktime):
        if not tmp:
            tmp.append(v)
        else:
            if abs(tmp[-1] - v) < 1:
                tmp.append(v)
            else:
                if len(tmp) >= 3:
                    bout_intervs.append([tmp[0], tmp[-1]])
                tmp = [v]
    if tmp and len(tmp) >= 3:
        bout_intervs.append([tmp[0], tmp[-1]])

    for x in bout_intervs:
        boutstart.append(x[0])
        boutend.append(x[1])

    # Convert starts to tracedata times
    for t in boutstart:
        ind = get_matched_time(time, t)
        bout_trace_start = time[ind]
        start_traces.append(bout_trace_start[0])

    # Convert ends to tracedata times
    for t in boutend:
        ind = get_matched_time(time, t)
        bout_trace_end = time[ind]
        end_traces.append(bout_trace_end[0])

    bouts_td_time = list(zip(start_traces, end_traces))

    bout_dff = []
    for i in range(1, nplot + 1):
        cell_traces = []
        for index, bout in enumerate(bouts_td_time):
            start = bout[0]
            end = bout[1]
            this_bout = (np.where((
                                          time > start) & (time < end))[0])
            bout_signal = np.mean((tracedata.iloc[this_bout, i]))
            cell_traces.append(bout_signal)
        bout_dff.append(np.mean(cell_traces))

    return start_traces, end_traces, bouts_td_time, bout_dff


def get_antibouts(tracedata, bouts_td_time, time):
    antibouts_dff = []
    antibouts_td_time = []
    k: int = 0
    cell: str

    for _ in tracedata.columns[1:]:
        k += 1
        cell_antibouts = []
        for i in range(len(bouts_td_time) - 1):
            nxt = i + 1

            this_bout = bouts_td_time[i]
            next_bout = bouts_td_time[nxt]
            start = this_bout[1]
            end = next_bout[0]
            # We want anti intervals longer than 10s
            if (end - start) > 10:
                # Get neuron spikes during anti-lick bout
                spikes_ind = list((np.where((time > start) & (time < end)))[0])
                cell_antibouts.append(np.mean(tracedata.iloc[spikes_ind, k]))
                if k == 1:
                    antibouts_td_time.append(time[spikes_ind])

        antibouts_dff.append(np.mean(cell_antibouts))

    return antibouts_dff, antibouts_td_time


def get_sponts(licktime, tracedata, time):
    idxs = np.where(np.diff(licktime) > 30)[0]
    spont_intervs = np.column_stack((licktime[idxs], licktime[idxs + 1]))
    sponts, stdevs = [], []
    print('yes')
    for cell in tracedata.columns[1:]:
        print(cell)
        cell_spont = []
        cell_sd = []
        for bound_low, bound_high in spont_intervs:
            ind0 = get_matched_time(time, bound_low)
            ind1 = get_matched_time(time, bound_high)
            cell_spont.append(
                np.mean(np.array(tracedata.loc[ind0[0]:ind1[0], cell])))
            cell_sd.append(
                np.std(np.array(tracedata.loc[ind0[0]:ind1[0], cell])))
        mean_spont = np.mean(cell_spont)
        sponts.append(mean_spont)
        mean_stdev = np.mean(cell_sd)
        stdevs.append(mean_stdev)

    return sponts, stdevs


def get_stats(tracedata, trial_times, time, session, raw_df):
    firstcell = tracedata.columns[1]
    trialstats = pd.DataFrame(columns=[
        'File', 'Cell', 'Stimulus',
        'Trial', 'Baseline (mean)',
        'Baseline (st_dev)',
        'Raw Signal (1s window)',
        'Signal Timestamps (start, stop)',
        'Baseline Timestamps (start, Stop)',
        'Shifted?',
        'deltaF/F',
        'Significant?'])

    summary = pd.DataFrame(columns=[
        'File', 'Stimulus', 'Num Trials'])

    count = 0
    for cell in tracedata.columns[1:]:
        count += 1

        for stim, times in trial_times.items():
            ntrials = len(times)
            avg_bls, avg_mags = [], []

            for iteration, trial in enumerate(times):

                # Index to analysis window
                data_ind = np.where(
                    (time > trial) & (time < trial + 5))[0]

                # Index to baseline window (current: 4s pre stim)
                bltime_ind = np.where(
                    (time > trial - 4) & (time < trial))[0]
                bltime = time[bltime_ind]
                bltime = [min(bltime), max(bltime)]

                # Baseline statistics
                bl_signal = np.array(tracedata.iloc[bltime_ind, count])
                mean_bl = np.mean(bl_signal)
                stdev_bl = np.std(bl_signal)
                c_interv = mean_bl + stdev_bl * 2.58

                # Get calcium trace & time
                signal = np.array(tracedata.iloc[data_ind, count])

                # Get peak signal & time
                peak_signal = max(signal)
                peak_ts = tracedata.loc[tracedata[cell]
                                        == peak_signal, 'Time(s)'].iloc[0]

                if peak_ts <= trial + 2.4:
                    peak_ts = max(bltime) + 2.4
                    shift = 'yes'
                else:
                    shift = 'no'

                # Get window 1s centered at peak
                peak_window_ind = get_peak_window(time, peak_ts)
                time_lower = time[peak_window_ind[0]]
                time_upper = time[peak_window_ind[1]]
                response_window = np.array(tracedata.loc[
                                           peak_window_ind[0]:peak_window_ind[1], cell])

                window_ts = [time_lower, time_upper]
                mean_mag = np.mean(response_window)
                dff = ((mean_mag - mean_bl) / mean_bl)

                # Output new Tracedata with Tastant trials only ##
                d = [stim, 'Trial {}'.format(iteration + 1)]
                d.extend(
                    ['-' for _ in range(len(tracedata.columns[2:]))])
                fill_id = pd.Series(d, index=tracedata.columns)

                # if DF/F is significant, output data
                if dff >= c_interv:
                    sig = 'Significant'
                    baseline_df = tracedata[tracedata['Time(s)'].between(
                        min(bltime), time_upper)]
                    raw_df.append(
                        fill_id, ignore_index=True)
                    raw_df.append(baseline_df)
                else:
                    sig = 'ns'

                # Check if peak_signal is a duplicate
                for _ in signal:
                    dup_check(signal, peak_signal)

                # Stats
                trial_stats = {
                    'File': session,
                    'Cell': cell,
                    'Stimulus': stim,
                    'Trial': 'Trial {}'.format(iteration + 1),
                    'Baseline (mean)': mean_bl,
                    'Baseline (st_dev)': stdev_bl,
                    'Signal Timestamps (start, stop)': window_ts,
                    'Baseline Timestamps (start, Stop)': bltime,
                    'Shifted?': shift,
                    'deltaF/F': dff,
                    'Significant?': sig
                }
                trialstats = trialstats.append(
                    trial_stats, ignore_index=True)
                avg_mags.append(mean_mag)
                avg_bls.append(mean_bl)

            if cell == firstcell:
                summary_dict = {
                    'File': session,
                    'Stimulus': stim,
                    'Num Trials': ntrials}
                summary = summary.append(
                    summary_dict, ignore_index=True)

    return trialstats, summary


def output(
        session, session_date, stats_dir,
        allstats, statsummary, lickstats, raw_df):
    if not os.path.isdir(stats_dir):
        os.mkdir(stats_dir)
    final_stats = pd.concat(allstats)

    final_summary = pd.concat(statsummary)
    final_lick = pd.concat(lickstats)
    print("Outputting data to Excel...")

    with pd.ExcelWriter(stats_dir + '/' + session_date + '_statistics.xlsx') as writer:
        final_summary.to_excel(
            writer, sheet_name='Summary', index=False)
        raw_df.to_excel(
            writer, sheet_name='Raw Data', index=False)
        final_lick.to_excel(
            writer, sheet_name='Lick cells', index=False)
        final_stats.to_excel(
            writer, sheet_name='Trial Stats', index=False)

    print(session + ' statistics successfully transfered to Excel!')

    return raw_df
