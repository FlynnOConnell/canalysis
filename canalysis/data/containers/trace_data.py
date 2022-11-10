#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# tracedata.py
Module (data.data_utils): Process traces exported from inscopix trace file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats

from canalysis.data.data_utils.file_handler import FileHandler


# %%


@dataclass
class TraceData:
    filehandler: FileHandler = FileHandler
    zscores: pd.DataFrame = None

    def __post_init__(self):
        self.tracedata = self.filehandler.get_tracedata()
        self._clean()
        # Core attributes
        self.signals = self._set_trace_signals()
        self.cells = np.asarray(self.tracedata.columns[1:])
        self.time = np.arange(0, self.tracedata.shape[0] / 10, 0.1)
        self.binsize = self.time[2] - self.time[1]
        self.zscores = self._get_zscores()

    def __repr__(self):
        return type(self).__name__

    @property
    def shape(self):
        return self.signals.shape

    def __hash__(self):
        return hash(repr(self))

    def _get_zscores(self) -> pd.DataFrame:
        zscores = pd.DataFrame(columns=self.signals.columns)
        for cell in self.signals.columns:
            allsignal = self.signals[cell]
            zscore = pd.Series(stats.zscore(allsignal))
            zscores[cell] = zscore
        zscores["time"] = self.time
        return zscores

    def _set_trace_signals(self) -> pd.DataFrame:
        temp = self.tracedata.copy()
        temp.pop("time")
        self.signals = temp
        return temp

    def _clean(self) -> None:

        # When CNMFe is run, we choose cells to "accept", but with manual ROI's every cell is accepted
        # so often that step is skipped. Need some way to check (check_if_accepted)
        _df = self.tracedata.copy()

        def check_if_accepted(_df):
            # If any cells marked as "accepted", use only those cells
            accepted_col = [col for col in _df.columns if " accepted" in col]
            return accepted_col

        accept = check_if_accepted(_df)
        if accept:
            accepted = np.where(_df.loc[0, :] == " accepted")[0]
            _df = _df.iloc[:, np.insert(accepted, 0, 0)]
        _df = _df.drop(0)
        _df = _df.rename(columns={" ": "time"})
        _df = _df.astype(float)
        _df = _df.reset_index(drop=True)
        _df.columns = [column.replace(" ", "") for column in _df.columns]
        _df["time"] = np.round(_df["time"], 2)
        self.tracedata = _df
        return None

    def reorder(self, cols) -> None:
        self.zscores = self.zscores[cols]
        self.zscores['time'] = self.time
