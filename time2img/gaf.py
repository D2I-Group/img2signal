#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from .utils.plotter import TimeSeriesPlotter
from pyts.image import GramianAngularField
from typing import Literal

class GAF_plotter(TimeSeriesPlotter):
    '''
    Gramian Angular Field (GAF) Plotter: Plots the Gramian Angular Field of a univariate time series signal.

    Usage:
    plotter = GAF_plotter()
    plotter.plot(x, method='summation', save_file='gaf.pdf', color_bar=False, label=True, save=True)

    Args:
    - x: np.ndarray
    - method: str
    - save_file: str
    - label: bool
    - save: bool
    - color_bar: bool
    '''
    def __init__(self):
        super().__init__()

    def plot(self, x: np.ndarray, method: Literal['summation', 'difference']='summation'):
        x = x.reshape(1, -1)
        transformer = GramianAngularField(method=method)
        gaf = transformer.fit_transform(x)
        return gaf[0]