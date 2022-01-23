#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 20:39:48 2022

@author: flynnoconnell
"""
import os

from statsmodels.sandbox.distributions.examples.ex_mvelliptical import md


class Singleton(type):
    """
    Singleton Class
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


