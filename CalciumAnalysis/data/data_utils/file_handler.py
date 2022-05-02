"""
#file_helpers.py

Module(util): File handling helper functions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# PROJECT_DIR = Path(__file__).parents[2]
# sys.path.append(
#     str(PROJECT_DIR / 'apps'))


class FileHandler(object):
    """
    File handler.
    Directory structure: 
        
    ./directory
        -| Animal 
        --| Date
        ---| Results
        -----| Graphs
        -----| Statistics
        ---| Data_traces*
        ---| Data_gpio_processed*
        Use tree() to print current directory tree.
        
    Args:
        directory (str): Path to directory containing data.
        animal_id (str): Animal name.
        session_date (str): Current session date
        tracename (str): String segment of trace to fetch.
        eventname (str): String segment of event file to fetch.
            -String used in unix-style pattern-matching, surrounded by wildcards:
                e.g. *traces*, *events*, *processed*, *GPIO*
            -Use these args to match string in file for processing.
    """

    def __init__(self,
                 directory: str,
                 animal_id: str,
                 session_date: str,
                 tracename: Optional[str] = 'traces',
                 eventname: Optional[str] = 'processed'):
        self.animal = animal_id
        self.date = session_date
        self._directory = Path(directory)
        self.session = self.animal + '_' + self.date
        self.animaldir = self._directory / self.animal
        self.sessiondir = self.animaldir / self.date
        self._tracename = tracename
        self._eventname = eventname
        self._make_dirs()

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, new_dir: str):
        self._directory = Path(new_dir)

    @property
    def tracename(self):
        return self._tracename

    @tracename.setter
    def tracename(self, new_tracename):
        self._tracename = new_tracename

    @property
    def eventname(self):
        return self._eventname

    @eventname.setter
    def eventname(self, new_eventname):
        self._eventname = new_eventname

    def tree(self):
        print(f'-|{self._directory}')
        for path in sorted(self.directory.rglob('[!.]*')):  # exclude .files
            depth = len(path.relative_to(self.directory).parts)
            spacer = '    ' * depth
            print(f'{spacer}-|{path.name}')
            return None

    def get_traces(self):
        tracefiles = self.sessiondir.rglob(f'*{self.tracename}*')
        for file in tracefiles:
            yield file

    def get_events(self):
        eventfiles = self.sessiondir.rglob(f'*{self.eventname}*')
        for file in eventfiles:
            yield file

    def get_tracedata(self):
        for filepath in self.get_traces():
            tracedata = pd.read_csv(filepath, low_memory=False)
            yield tracedata

    def get_eventdata(self):
        for filepath in self.get_events():
            eventdata = pd.read_csv(filepath, low_memory=False)
            yield eventdata

    def unique_path(self, filename):
        counter = 0
        while True:
            counter += 1
            path = self.sessiondir / filename
            if not path.exists():
                return path

    def _make_dirs(self):
        path = self.sessiondir
        Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        return None

    def get_cwd(self):
        return str(self._directory.cwd())


    def get_home_dir(self):
        return str(self._directory.home())
