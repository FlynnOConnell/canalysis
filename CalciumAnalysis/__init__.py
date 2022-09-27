"""A package for analysis, statistics and visualization of Calcium Imaging data,
specifically from including from Inscopix."""

from __future__ import annotations
import logging
import faulthandler
import calcium_data
from containers.taste_data import TasteData
from containers.all_data import AllData
from containers.gpio_data import GpioData
from containers.event_data import EventData
from containers.eating_data import EatingData
from containers.trace_data import TraceData
faulthandler.enable()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

__all__ = [TasteData, EatingData, AllData, GpioData, EventData, TraceData, calcium_data]

