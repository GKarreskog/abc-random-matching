#%%
%run random_match_popfuns.py
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ABC stuff
from pyabc.visualization import plot_kde_matrix
from pyabc.visualization import plot_kde_1d

import os
import tempfile

import scipy.stats as st
import scipy as scp


from pyabc import (ABCSMC, RV, Distribution,
                   PercentileDistanceFunction)

import pyabc
