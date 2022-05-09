#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Namespace __init__.py

CalciumAnalysis package for processing acquired calcium traces and corresponding events.
"""
from pathlib import Path
import yaml

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

my_path = Path(__file__).parent.resolve()  # resolve to get rid of any symlinks
config_path = my_path.parent / 'config.yaml'

with config_path.open() as config_file:
    config = yaml.safe_load(config_file)
