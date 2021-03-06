#!/usr/bin/python3

import copy

import joblib
from matplotlib import gridspec
from matplotlib import lines
from matplotlib import pyplot
from matplotlib import ticker
import numpy
import seaborn

import common
import growth_rates
import parameters


save = True

figsize = (6, 5.5)


def _run_one(p, param0, param1, dP0, dP1):
    p = copy.copy(p)
    setattr(p, param0, dP0)
    setattr(p, param1, dP1)
    return growth_rates.get_growth_rate(p)


def build():
    nparamsets = len(parameters.parameter_sets)
    nparams = len(common.sensitivity_parameters)
    r0 = numpy.ones((nparamsets,
                     nparams - 1, nparams - 1,
                     common.npoints, common.npoints))
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        for (k, paramset) in enumerate(parameters.parameter_sets.items()):
            p_name, p = paramset
            print('Running parameter set {}.'.format(p_name))
            for i in range(nparams - 1):
                param0, param0_name = common.sensitivity_parameters[i + 1]
                param0baseline = getattr(p, param0)
                dPs0 = common.get_dPs(param0, param0baseline)
                for j in range(i + 1):
                    param1, param1_name = common.sensitivity_parameters[j]
                    param1baseline = getattr(p, param1)
                    dPs1 = common.get_dPs(param1, param1baseline)
                    print('\tRunning {} and {}.'.format(param0, param1))
                    r0_ = parallel(joblib.delayed(_run_one)(p, param0, param1,
                                                            dP0, dP1)
                                   for dP0 in dPs0
                                   for dP1 in dPs1)
                    r0[k, i, j] = numpy.asarray(r0_).reshape((len(dPs0), -1))
    return r0


def _hide_ticklabels(ax):
    for l in ax.get_ticklabels():
        l.set_visible(False)
    ax.offsetText.set_visible(False)


def _format_axis(ax, param0_name, param1_name, left, right, top, bottom,
                 fontsize):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
        axis.set_minor_locator(ticker.NullLocator())
        scale = axis.get_scale()
        if scale in ('linear', 'logit'):
            axis.set_major_locator(ticker.FixedLocator(
                [0.1, 0.3, 0.5, 0.7, 0.9]))
        elif scale == 'log':
            axis.set_major_locator(ticker.LogLocator(subs = [1, 2, 5]))

    ax.tick_params(top = True, bottom = True, right = True, left = True,
                   labelsize = fontsize)

    if top:
        ax.tick_params(labeltop = True, labelbottom = False)
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(param1_name, fontsize = fontsize)
    if bottom:
        ax.set_xlabel(param1_name, fontsize = fontsize)
    if not (top or bottom):
        _hide_ticklabels(ax.xaxis)

    if left:
        ax.set_ylabel(param0_name, fontsize = fontsize)
    if right:
        ax.tick_params(labelright = True, labelleft = False)
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(param0_name, fontsize = fontsize)
    if not (left or right):
        _hide_ticklabels(ax.yaxis)

    # Stupid bug in matplotlib.
    if ax.get_xscale() == 'logit':
        for loc in ('bottom', 'top'):
            ax.spines[loc]._adjust_location()
    if ax.get_yscale() == 'logit':
        for loc in ('left', 'right'):
            ax.spines[loc]._adjust_location()


def plot(r0):
    fontsize = 8

    # Contour levels every log10.
    log_rgr = numpy.log10(r0)
    log_rgr_absmax = numpy.max(numpy.abs(log_rgr))
    log_rgr_levels = numpy.linspace(- numpy.ceil(log_rgr_absmax),
                                    numpy.ceil(log_rgr_absmax),
                                    2 * int(numpy.ceil(log_rgr_absmax)) + 1)
    contour_levels = 10 ** log_rgr_levels

    colors = seaborn.color_palette()

    nparams = len(common.sensitivity_parameters)
    nrows = ncols = nparams
    gs = gridspec.GridSpec(nrows, ncols)
    fig = pyplot.figure(figsize = figsize)
    for row in range(nrows):
        for col in range(ncols):
            if row != col:
                param0, param0_name = common.sensitivity_parameters[row]
                param1, param1_name = common.sensitivity_parameters[col]
                yscale = common.get_scale(param0)
                xscale = common.get_scale(param1)
                ax = fig.add_subplot(gs[row, col],
                                     xscale = xscale,
                                     yscale = yscale)
                for (k, n) in enumerate(common.parameter_sets_ordered):
                    p = parameters.parameter_sets[n]
                    param0baseline = getattr(p, param0)
                    param1baseline = getattr(p, param1)
                    y = common.get_dPs(param0, param0baseline)
                    x = common.get_dPs(param1, param1baseline)
                    X, Y = numpy.meshgrid(x, y)
                    if row > col:
                        i = row - 1
                        j = col
                        Z = r0[k, i, j]
                    elif row < col:
                        i = col - 1
                        j = row
                        Z = r0[k, i, j].T
                    else:
                        raise RuntimeError
                    cs = ax.contour(X, Y, Z,
                                    contour_levels,
                                    colors = [colors[k]],
                                    alpha = common.alpha,
                                    linestyles = 'solid')
                    ax.clabel(cs,
                              inline = True,
                              fmt = '%g d$^{-1}$',
                              fontsize = fontsize,
                              colors = [colors[k] + (common.alpha, )])
                    # We only need to draw these once.
                    if k == 0:
                        ax.axvline(param1baseline, **common.baseline_style)
                        ax.axhline(param0baseline, **common.baseline_style)
                left = (col == 0)
                # left = (col == 0) or (row == 0 and col == 1)
                right = (col == ncols - 1)
                top = (row == 0)
                bottom = (row == nrows - 1)
                _format_axis(ax, param0_name, param1_name, left, right,
                             top, bottom, fontsize)
    fig.align_ylabels()
    handles = [lines.Line2D([], [], color = c, alpha = common.alpha)
               for c in seaborn.color_palette()]
    labels = common.parameter_sets_ordered
    leg = fig.legend(handles, labels,
                     loc = (0.74, 0.153),
                     frameon = False,
                     fontsize = fontsize + 2)
    fig.tight_layout()
    if save:
        common.savefig(fig)
    return fig


if __name__ == '__main__':
    r0 = common.load_or_build_data(build)
    plot(r0)
    pyplot.show()
