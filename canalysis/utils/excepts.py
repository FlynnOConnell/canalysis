#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 09:17:47 2022

@author: flynnoconnell
"""


class DataFrameError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "DataFameError has been raised"


class ComponentError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Number components was chosen to be larger than the total number of cells (features)."


class DuplicateError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "A duplicate time may have been chosen for time of peak"


class ParameterError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Only rbf or linear kernals allowed for this analysis."


class MergeError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Cant merge these indices."


class FileError(Exception):
    def __init__(self, msg: str, filetype: str = "Traces"):
        if not msg:
            raise AttributeError("Message param missing from FileError call")
        else:
            self.message = msg
            self.filetype = filetype

    def __str__(self):
        return f"{self.message} - File type: {self.filetype}"


class MatchError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Time to match (match) larger than iterable to match to, likely wrong order of match/time."


class PCAError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "DataFrame not properly set, missing column for 'colors' or 'events' .. "
