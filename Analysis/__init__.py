#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 20:39:48 2022

@author: flynnoconnell
"""
import os
from module.logging_config_manager import setup_logging
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

class Singleton(type):
    """
    Singleton Class
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]



setup_logging(default_path=os.path.join("/".join(__file__.split('/')[:-1]), 'config', 'module_logging.yaml'))

# todo: Prathyush SP - Debug code breakages due to changing start method
# import multiprocessing
# multiprocessing.context._force_start_method('spawn')

from module.metadata import metadata as md

__version__ = md.__version__

if __name__ == '__main__':
    print('module.__init__ success . . .')