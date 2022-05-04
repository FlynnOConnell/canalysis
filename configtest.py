# -*- coding: utf-8 -*-
import yaml

with open('defaults.yml', 'r') as c:
    config = yaml.safe_load(c)

datadir = config['DIRS']
