#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#process_data.py

Module(analysis): Stats class for PCA and general statistics/output.
"""
from __future__ import annotations

from typing import Iterable, Any, ClassVar
import pandas as pd

from utils import funcs


def sort_by_value(data):
    pass


class ProcessData:
    def __init__(self, data,):
        self.data: ClassVar = data
        self.signals: pd.DataFrame = data.tracedata.signals
        self.zscores: pd.DataFrame = data.tracedata.zscores
        self.time: pd.Series | Iterable[Any] = data.tracedata.time
        self.cells: Iterable[Any] = data.tracedata.cells
        self.avgs: dict = data.nr_avgs
        self.session: str = data.filehandler.session
        self.trial_times: dict = data.eventdata.trial_times
        self.timestamps: dict = data.eventdata.timestamps
        self.antibouts = self.get_antibouts()
        self.sponts = self.get_sponts()

    def get_antibouts(self) -> pd.DataFrame | pd.Series:
        antibouts = pd.DataFrame()
        for interv in funcs.interval(self.timestamps["Lick"], gap=10, outer=True):
            df = self.signals.loc[(self.time > (interv[0])) & (self.time < (interv[1]))]
            antibouts = pd.concat([antibouts, df], axis=0)
        return antibouts

    def get_sponts(self) -> pd.DataFrame | pd.Series:
        sponts = pd.DataFrame()
        for interv in funcs.interval(self.timestamps["Lick"], gap=30, outer=True):
            df = self.signals[(self.time > (interv[0])) & (self.time < (interv[1]))]
            sponts = pd.concat([sponts, df], axis=0)
        return sponts
