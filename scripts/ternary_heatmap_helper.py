# -*- coding: utf-8 -*-
""" A helper function to ternary plot.
"""
#
# written by Chengcheng Shao <sccotte@gmail.com>

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ternary.helpers import simplex_iterator


def heatmap_density(X, Y, Z, scale, boundary=True):
    """
    A function to calculate the how many point located in bin (x, y, z)
    It is very sutiable for hexgonal bin
    """
    X = np.around(X).astype(int)
    Y = np.around(Y).astype(int)
    Z = np.around(Z).astype(int)
    data = dict()
    for i, j, k in simplex_iterator(scale=scale, boundary=boundary):
        xb = (X == i)
        yb = (Y == j)
        zb = (Z == k)
        data[(i, j)] = (xb & yb & zb).sum()
    return data


def colorbar_hack(ax,
                  vmin,
                  vmax,
                  cmap,
                  log_norm=False,
                  scientific=False,
                  cbarlabel=None):
    """
    Inhanced Colorbar hack to insert colorbar on ternary plot.

    Parameters
    ----------
    vmin: float
        Minimum value to portray in colorbar
    vmax: float
        Maximum value to portray in colorbar
    cmap: Matplotlib colormap
        Matplotlib colormap to use
    log_norm: boolean,
        use LogNorm to norm the colorcmap, make sure vmin and vmax are
        sutiable for log

    """
    # http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    if log_norm is True:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # Fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cb = plt.colorbar(sm, ax=ax)
    if cbarlabel is not None:
        cb.set_label(cbarlabel)
    if scientific:
        cb.locator = mpl.ticker.LinearLocator(numticks=7)
        cb.formatter = mpl.ticker.ScalarFormatter()
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
