"""A package for analysis, statistics and visualization of Calcium Imaging data,
specificially from including from Inscopix."""

from __future__ import annotations
import logging
import faulthandler

from data.taste_data import TasteData
from data.all_data import AllData
from data.gpio_data import GpioData
from data.event_data import EventData
from data.eating_data import EatingData
from data.trace_data import TraceData
faulthandler.enable()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

__all__ = [TasteData, EatingData, AllData, GpioData, EventData, TraceData]
print("Importing", __name__)
