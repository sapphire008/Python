"""Unit tests for the viewdata module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import time
import tempfile

import numpy as np

from spikedetekt2.dataio import *
from klustaviewa.views.viewdata import *
from klustaviewa.views.tests.utils import show_view
from klustaviewa.views import WaveformView, FeatureView


with Experiment('n6mab041109_60sec_n6mab031109_MKKdistfloat_25_regular100_1', dir='data') as exp:
    chgrp = exp.channel_groups[0]
    data = get_waveformview_data(exp, clusters=[0])
    show_view(WaveformView, **data)
    