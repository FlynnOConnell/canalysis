"""A package for analysis, statistics and visualization of Calcium Imaging data,
specificially from including from Inscopix."""

from __future__ import annotations
import logging
import faulthandler

faulthandler.enable()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
