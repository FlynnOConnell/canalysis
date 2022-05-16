"""
#file_helpers.py

Module(misc/file_helpers): File handling data-container class to keep all file-related data.
"""
from __future__ import annotations
from collections import namedtuple
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from misc import funcs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


class FileHandler:
    """
    File handler. From directory, return genorator looping through files.

    Parameters:
    ___________
    animal: str
        - Animal ID, letter + number combo
    date: str
        - Session date, all_numeric
    _directory: str
        - Path as str, contains /
    _tracename: Optional[str]
        - Pattern matching to apply for trace files
    _eventname: Optional[str]
        - Pattern matching to apply for event files

    Directory structure:
    ___________
    
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

    def __init__(self,
                 animal,
                 date,
                 _dir,
                 _tracename: Optional[str] = 'traces',
                 _eventname: Optional[str] = 'processed',
                 _gpioname: Optional[str] = 'gpio.csv'
                 ) -> None:

        self.animal = animal
        self.date = date
        self._directory: Path = Path(_dir)
        self._tracename: Optional[str] = _tracename
        self._eventname: Optional[str] = _eventname
        self._gpioname: Optional[str] = _gpioname
        self.color_dict: namedtuple
        self._validate()

        self.session: Path = Path(self.animal + self.date)
        self.animaldir: Path = Path(self._directory / self.animal)
        self.sessiondir: Path = Path(self.animaldir / self.date)
        self._gpio_file: Optional[bool] = False
        self._make_dirs()

    def _validate(self):
        """Validate format of input data. Raises AttributeErrors for each check."""
        if not funcs.check_numeric(self.date):
            raise AttributeError(f'Date must be all numeric, not {self.date}')
        if not funcs.check_path(self._directory):
            raise AttributeError(f'Directory must contain "/", not {self._directory}')
        if funcs.check_numeric(self.animal) or funcs.check_path(self.animal):
            raise AttributeError(f'Animal must not be only numeric or contain path characters, {self.animal}')

    @property
    def directory(self) -> Path:
        return self._directory

    @directory.setter
    def directory(self, new_dir: str) -> None:
        self._directory: str = new_dir

    @property
    def tracename(self) -> str:
        return self._tracename

    @tracename.setter
    def tracename(self, new_tracename: str) -> None:
        self._tracename: str = new_tracename

    @property
    def eventname(self) -> str:
        return self._eventname

    @eventname.setter
    def eventname(self, new_eventname: str) -> None:
        self._eventname: str = new_eventname

    @property
    def gpioname(self) -> str:
        return self._gpioname

    @gpioname.setter
    def gpioname(self, new_gpioname: str):
        self._gpioname: str = new_gpioname

    def get_traces(self) -> list[Path]:
        return [p for p in self.sessiondir.glob(f'*{self._tracename}*')]

    def get_events(self) -> list[Path]:
        return [p for p in self.sessiondir.glob(f'*{self._eventname}*')]

    def get_gpio_files(self) -> list[Path]:
        return [p for p in self.sessiondir.glob(f'*{self._gpioname}*')]

    def get_tracedata(self) -> pd.DataFrame:
        tracefiles: list[Path] = self.get_traces()
        if tracefiles is None:
            raise FileNotFoundError(f'No files in {self.sessiondir} matching "{self._tracename}"')
        if len(tracefiles) > 1:
            logging.info(f'Multiple trace-files found in {self.sessiondir} matching "{self._tracename}":')
            for tracefile in tracefiles:
                logging.info(f'{tracefile.stem}')
        logging.info(f'Taking file: {tracefiles[0].stem}')
        return pd.read_csv(str(tracefiles[0]), low_memory=False)

    def get_eventdata(self) -> pd.DataFrame:
        eventfiles: list[Path] = self.get_events()
        if eventfiles is None:
            raise FileNotFoundError(f'No files in {self.sessiondir} matching "{self._eventname}"')
        if len(eventfiles) > 1:
            logging.info(f'Multiple event-files found in {self.sessiondir} matching "{self._eventname}":')
            for event_file in eventfiles:
                logging.info(f'{event_file}')
            logging.info(f'Taking file: {eventfiles[0]}')
        return pd.read_csv(str(eventfiles[0]), low_memory=False)

    def get_gpiodata(self) -> pd.DataFrame:
        gpiofiles: list[Path] = self.get_gpio_files()
        if gpiofiles is None:
            raise FileNotFoundError(f'No files in {self.sessiondir} matching "{self._gpioname}"')
        if len(gpiofiles) > 1:
            self._gpio_file = True
            logging.info(f'Multiple gpio-files found in {self.sessiondir} matching "{self._gpioname}":')
            for gpio_file in gpiofiles:
                logging.info(f'{gpio_file}')
            logging.info(f'Taking file: {gpiofiles[0]}')
        return pd.read_csv(str(gpiofiles[0]), low_memory=False)

    def unique_path(self, filename) -> Path:
        counter = 0
        while True:
            counter += 1
            path = self.sessiondir / filename
            if not path.exists():
                return path

    def _make_dirs(self) -> None:
        self.sessiondir.parents[0].mkdir(parents=True, exist_ok=True)
        return None

    def get_cwd(self) -> str:
        return str(self._directory.cwd())

    def get_home_dir(self) -> str:
        return str(self._directory.home())

    def tree(self) -> None:
        print(f'-|{self._directory}')
        for path in sorted(self.sessiondir.rglob('[!.]*')):  # exclude .files
            depth = len(path.relative_to(self.sessiondir).parts)
            spacer = '    ' * depth
            print(f'{spacer}-|{path.name}')
            return None
