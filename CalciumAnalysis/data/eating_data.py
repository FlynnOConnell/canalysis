#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#eating_data.py

Module: Classes for food-related data processing.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
from .data_utils.file_handler import FileHandler

logger = logging.getLogger(__name__)


@dataclass
class EatingData:
    filehandler: FileHandler = FileHandler
    adjust: Optional[int] = None
    split: Optional[bool] = False

    def __post_init__(self,):
        self.eatingdata = self.filehandler.get_eatingdata().sort_values("TimeStamp")
        # Core attributes
        self.__clean()
        self.set_adjust()

    def __hash__(self, ):
        return hash(repr(self))

    def set_adjust(self,) -> None:
        for column in self.eatingdata.columns[1:]:
            self.eatingdata[column] = self.eatingdata[column] + self.adjust

    def __clean(self, ) -> None:
        self.eatingdata.drop(
            ["Marker Type", "Marker Event Id", "Marker Event Id 2", "value1", "value2"],
            axis=1,
            inplace=True,
        )
        self.eatingdata = self.eatingdata.loc[
            self.eatingdata["Marker Name"].isin(["Entry", "Eating", "Grooming"])
        ]


