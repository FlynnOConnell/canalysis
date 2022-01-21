# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:55:34 2022

@author: flynnoconnell
"""
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
import statistics
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


@dataclass
class Traces:
    data: str
    tracedata: pd.DataFrame
    tracedata_out: pd.DataFrame
    trc: pd.DataFrame

    @staticmethod
    def PopData(self, tracedata):
        acceptcols = np.where(tracedata.loc[0, :] == ' accepted')[0]
        tracedata = tracedata.iloc[:, np.insert(acceptcols, 0, 0)]
        tracedata = tracedata.drop(0)
        tracedata = tracedata.rename(columns={' ': 'Time(s)'})
        tracedata = tracedata.astype(float)
        tracedata = tracedata.reset_index(drop=True)
        tracedata.columns = [column.replace(' ', '')
                             for column in tracedata.columns]
        tracedata_out = pd.DataFrame(data=None, columns=tracedata.columns)
        trc = (tracedata.drop('Time(s)', axis=1))  # Used in PCA analysis

        return tracedata, tracedata_out, trc

    @staticmethod
    def getPeakWindow(time, peak_time):
        """ Returns the index of tracedata centered 1s around the peak flourescent value for
            that trial.
        """
        aux, window_ind = [], []
        for valor in time:
            aux.append(abs(peak_time - valor))

        window_ind.append(aux.index(min(aux)) - 20)
        window_ind.append(aux.index(min(aux)) + 20)

        return window_ind

    def getMatchedTime(time, *argv):
        """  Finds the closest number in tracedata time to the input.
        """
        return_index = []
        for arg in argv:
            temp = []
            for valor in time:
                temp.append(abs(arg - valor))
            return_index.append(temp.index(min(temp)))
        return return_index


@dataclass
class Events:
    data: str()
    eventdata: pd.DataFrame

    def PopEvents(eventdata):
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
                        next = tslist[i + 1]
                        tslist = np.delete(tslist, i + 1)
                        print(f"Deleted timestamp {next}")
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


@dataclass
class Lick():
    lickstats: pd.DataFrame()
    bout_dff: list()
    antibout_dff: list()
    antibout_td_time: list()
    bout_td_time: list()
    tmp: list()
    boutstart: list()
    boutend: list()
    start_traces: list()
    end_traces: list()

    def getBout(licktime, tracedata, time, nplot):
        #### Make bout intervals ----------------
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
        for t in (boutstart):
            ind = Traces.getMatchedTime(time, t)
            bout_trace_start = time[ind]
            start_traces.append(bout_trace_start[0])

        # Convert ends to tracedata times
        for t in (boutend):
            ind = Traces.getMatchedTime(time, t)
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

    def getAntiBout(tracedata, bouts_td_time, time):
        antibout_dff = []
        antibouts_td_time = []
        ii = 0
        for cell in tracedata.columns[1:]:
            ii += 1
            cell_antibouts = []
            for i in range(0, len(bouts_td_time) - 1):
                nxt = i + 1

                this_bout = bouts_td_time[i]
                next_bout = bouts_td_time[nxt]
                start = this_bout[1]
                end = next_bout[0]
                # We want anti intervals longer than 10s
                if (end - start) > 10:
                    # Get neuron spikes during anti-lick bout
                    spikes_ind = list((np.where((time > start) & (time < end)))[0])
                    cell_antibouts.append(np.mean(tracedata.iloc[spikes_ind, ii]))
                    if ii == 1:
                        antibouts_td_time.append(time[spikes_ind])

            antibout_dff.append(np.mean(cell_antibouts))

        return antibout_dff

    def getSpont(licktime, tracedata, time):
        idxs = np.where(np.diff(licktime) > 30)[0]
        spont_intervs = np.column_stack((licktime[idxs], licktime[idxs + 1]))
        sponts, stdevs = [], []
        print('yes')
        for cell in tracedata.columns[1:]:
            print(cell)
            cell_spont = []
            cell_sd = []
            for bound_low, bound_high in spont_intervs:
                ind0 = Traces.getMatchedTime(time, bound_low)
                ind1 = Traces.getMatchedTime(time, bound_high)
                cell_spont.append(
                    np.mean(np.array(tracedata.loc[ind0[0]:ind1[0], cell])))
                cell_sd.append(
                    np.std(np.array(tracedata.loc[ind0[0]:ind1[0], cell])))
            mean_spont = np.mean(cell_spont)
            sponts.append(mean_spont)
            mean_stdev = np.mean(cell_sd)
            stdevs.append(mean_stdev)

        return sponts, stdevs

    def Stats(tracedata, trial_times, time, tracedata_out, session):
        firstcell = tracedata.columns[1]
        trialStats = pd.DataFrame(columns=[
            'File', 'Cell', 'Stimulus',
            'Trial', 'Baseline (mean)',
            'Baseline (st_dev)',
            'Raw Signal (1s window)',
            'Signal Timestamps (start, stop)',
            'Baseline Timestamps (start, Stop)',
            'Shifted?',
            'deltaF/F',
            'Significant?'])

        tastantStats = pd.DataFrame(columns=[
            'File', 'Cell', 'Stimulus',
            'Trials', 'Baselines (mean)',
            'Responses (mean)'])

        summary = pd.DataFrame(columns=[
            'File', 'Stimulus', 'Num Trials'])

        count = 0
        for cell in tracedata.columns[1:]:
            count += 1
            trial_stats = {}
            for stim, times in trial_times.items():
                ntrials = len(times)
                avg_bls, avg_mags = [], []

                for iteration, trial in enumerate(times):

                    # Index to analysis window
                    data_ind = np.where(
                        (time > trial) & (time < trial + 5))[0]
                    this_time = time[data_ind]

                    # Index to baseline window (current: 4s pre stim)
                    BLtime_ind = np.where(
                        (time > trial - 4) & (time < trial))[0]
                    BLtime = time[BLtime_ind]
                    bltime = [min(BLtime), max(BLtime)]

                    # Baseline statistics
                    BL_signal = np.array(tracedata.iloc[BLtime_ind, count])
                    mean_bl = np.mean(BL_signal)
                    stdev_bl = np.std(BL_signal)
                    c_interv = mean_bl + stdev_bl * 2.58

                    # Get calcium trace & time
                    signal = np.array(tracedata.iloc[data_ind, count])

                    # Get peak signal & time
                    peak_signal = max(signal)
                    peak_ts = tracedata.loc[tracedata[cell]
                                            == peak_signal, 'Time(s)'].iloc[0]

                    if peak_ts <= trial + 2.4:
                        peak_ts = max(BLtime) + (2.4)
                        data_ind = np.where(
                            time == peak_ts)[0]
                        this_time = time[data_ind]
                        shift = 'yes'
                    else:
                        shift = 'no'

                    # Get window 1s centered at peak

                    peak_window_ind = Traces.getPeakWindow(peak_ts)

                    time_lower = time[peak_window_ind[0]]
                    time_upper = time[peak_window_ind[1]]

                    response_window = np.array(tracedata.loc[
                                               peak_window_ind[0]:peak_window_ind[1], cell])

                    window_ts = [time_lower, time_upper]
                    mean_mag = np.mean(response_window)

                    dFF = ((mean_mag - mean_bl) / mean_bl)

                    # Output new Tracedata with Tastant trials only ##
                    d = [stim, 'Trial {}'.format(iteration + 1)]
                    d.extend(
                        ['-' for i in range(len(tracedata.columns[2:]))])
                    iD = pd.Series(d, index=tracedata.columns)

                    # if DF/F is significant, output data
                    if dFF >= c_interv:
                        sig = 'Significant'
                        baseline_df = tracedata[tracedata['Time(s)'].between(
                            min(BLtime), time_upper)]
                        tracedata_out = tracedata_out.append(
                            iD, ignore_index=True)
                        tracedata_out = tracedata_out.append(baseline_df)
                    else:
                        sig = 'ns'

                    # Check if peak_signal is a duplicate
                    for value in signal:
                        dup_check(signal, peak_signal)

                    ## Stats ##
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
                        'deltaF/F': dFF,
                        'Significant?': sig
                    }
                    trialStats = trialStats.append(
                        trial_stats, ignore_index=True)
                    avg_mags.append(mean_mag)
                    avg_bls.append(mean_bl)

                wilcox_dict = {
                    'File': session,
                    'Cell': cell,
                    'Stimulus': stim,
                    'Trials': ntrials,
                    'Baselines (mean)': statistics.mean(avg_bls),
                    'Responses (mean)': statistics.mean(avg_mags)
                }
                tastantStats = tastantStats.append(
                    wilcox_dict, ignore_index=True)
                if cell == firstcell:
                    summary_dict = {
                        'File': session,
                        'Stimulus': stim,
                        'Num Trials': ntrials}
                    summary = summary.append(
                        summary_dict, ignore_index=True)

        return trialStats, summary


def output(session, session_date, stats_dir, allstats, statsummary, lickstats, tracedata_out):
    if not os.path.isdir(stats_dir):
        os.mkdir(stats_dir)
    final_stats = pd.concat(allstats)

    final_summary = pd.concat(statsummary)
    final_lick = pd.concat(lickstats)
    print("Outputting data to Excel...")

    with pd.ExcelWriter(stats_dir + '/' + session_date + '_statistics.xlsx') as writer:
        final_summary.to_excel(
            writer, sheet_name='Summary', index=False)
        tracedata_out.to_excel(
            writer, sheet_name='Raw Data', index=False)
        final_lick.to_excel(
            writer, sheet_name='Lick cells', index=False)
        final_stats.to_excel(
            writer, sheet_name='Trial Stats', index=False)

    print(session + ' statistics successfully transfered to Excel!')
