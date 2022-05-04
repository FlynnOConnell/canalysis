"""
#file_helpers.py

Module(util): File handling helper functions.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator, Any

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

PROJECT_DIR = Path(__file__).parents[2]
sys.path.append(
    str(PROJECT_DIR / 'apps'))


@dataclass
class FileHandler:
    """
    File handler. From directory, return genorator looping through files.
        Directory structure:
    ./directory
        -| Animal
        --| Date
        ---| Results
        -----| Graphs
        -----| Statistics
        ---| Data_traces*
        ---| Events_gpio (or)
        ---| Events_gpio_processed*
        Use tree() to print current directory tree.

    """
    # Dataclass Fields
    _directory: str = field(repr=False)
    animal: str = field(repr=False)
    date: str = field(repr=False)
    _tracename: Optional[str] = 'traces'
    _eventname: Optional[str] = 'processed'
    gpio_file: Optional[bool] = False
    _gpioname: Optional[str] = 'gpio'

    def __post_init__(self):
        self._directory = Path(self._directory)
        self.session: str = f'{self.animal}_{self.date}'
        self.animaldir: str = self._directory / self.animal
        self.sessiondir: str = self.animaldir / self.date
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
        for path in sorted(self.sessiondir.rglob('[!.]*')):  # exclude .files
            depth = len(path.relative_to(self.sessiondir).parts)
            spacer = '    ' * depth
            print(f'{spacer}-|{path.name}')
            return None

    # Generators to iterate each file matching pattern
    def get_traces(self):
        tracefiles = self.sessiondir.rglob(f'*{self.tracename}*')
        for file in tracefiles:
            yield file

    def get_events(self):
        eventfiles = self.sessiondir.rglob(f'*{self.eventname}*')
        for file in eventfiles:
            yield file

    def get_gpio_files(self) -> Generator[Any, Any, None]:
        gpiofile = self.sessiondir.rglob(f'*{self._gpioname}')
        for file in gpiofile:
            yield file

    # Generators to iterate each matched file and convert to pd.DataFrame 
    def get_tracedata(self):
        for filepath in self.get_traces():
            tracedata = pd.read_csv(filepath, low_memory=False)
            yield tracedata

    def get_eventdata(self):
        for filepath in self.get_events():
            eventdata = pd.read_csv(filepath, low_memory=False)
            yield eventdata

    def get_gpiodata(self) -> Generator[Any, Any, None]:
        filepath: str
        for filepath in self.get_gpio_files:
            gpiodata = pd.read_csv(filepath, low_memory=False)
            yield gpiodata

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
