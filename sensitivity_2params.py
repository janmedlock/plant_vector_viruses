#!/usr/bin/python3

import copy

import joblib
from matplotlib import gridspec
from matplotlib import lines
from matplotlib import pyplot
from matplotlib import ticker
import numpy

import common
import growth_rates
import parameters

import seaborn_quiet as seaborn


save = True

figsize = (8.5, 8)
seaborn.set_palette('Dark2')
alpha = 0.7


def _run_one(p, param0, param1, dP0, dP1):
    p = copy.copy(p)
    setattr(p, param0, dP0)
    setattr(p, param1, dP1)
    return growth_rates.get_growth_rate(p)


def build():
    nparamsets = len(parameters.parameter_sets)
    nparams = len(common.sensitivity_parameters)
    rel_growth_rate = numpy.ones((nparamsets,
                                  nparams - 1, nparams - 1,
                                  common.npoints,
                                  common.npoints))
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        for (k, p) in enumerate(parameters.parameter_sets.values()):
            r0baseline = growth_rates.get_growth_rate(p)
            for i in range(nparams - 1):
                for j in range(i + 1):
                    param0, param0_name = common.sensitivity_parameters[i + 1]
                    param1, param1_name = common.sensitivity_parameters[j]
                    param0baseline = getattr(p, param0)
                    param1baseline = getattr(p, param1)
                    dPs0 = common.get_dPs(param0, param0baseline)
                    dPs1 = common.get_dPs(param1, param1baseline)
                    r0 = parallel(joblib.delayed(_run_one)(p, param0, param1,
                                                           dP0, dP1)
                                 for dP0 in dPs0
                                 for dP1 in dPs1)
                    r0 = numpy.asarray(r0).reshape((len(dPs0), -1))
                    rel_growth_rate[k, i, j] = r0 / r0baseline
    return rel_growth_rate


def _hide_ticklabels(ax):
    for l in ax.get_ticklabels():
        l.set_visible(False)
    ax.offsetText.set_visible(False)


def _format_axis(ax, xlabel, ylabel):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
        scale = axis.get_scale()
        if scale == 'linear':
            axis.set_major_locator(ticker.MaxNLocator(nbins = 4))
        elif scale == 'log':
            axis.set_major_locator(ticker.LogLocator(subs = [1, 2, 5]))
            axis.set_minor_locator(ticker.NullLocator())

    ax.tick_params(labelsize = 9)

    if ax.is_last_row():
        ax.set_xlabel(xlabel, fontsize = 9)
    else:
        _hide_ticklabels(ax.xaxis)

    if ax.is_first_col():
        ax.set_ylabel(ylabel, fontsize = 9)
    else:
        _hide_ticklabels(ax.yaxis)


def plot(rel_growth_rate):
    # Contour levels every log10.
    log_rgr = numpy.log10(rel_growth_rate)
    log_rgr_absmax = numpy.max(numpy.abs(log_rgr))
    log_rgr_levels = numpy.linspace(- numpy.ceil(log_rgr_absmax),
                                    numpy.ceil(log_rgr_absmax),
                                    2 * int(numpy.ceil(log_rgr_absmax)) + 1)
    contour_levels = 10 ** log_rgr_levels

    colors = seaborn.color_palette()

    nparams = len(common.sensitivity_parameters)
    nrows = nparams - 1
    ncols = nparams - 1
    gs = gridspec.GridSpec(nrows, ncols)
    fig = pyplot.figure(figsize = figsize)
    for row in range(nrows):
        for col in range(row + 1):
            param0, param0_name = common.sensitivity_parameters[row + 1]
            param1, param1_name = common.sensitivity_parameters[col]

            yscale = common.get_scale(param0)
            xscale = common.get_scale(param1)

            ax = fig.add_subplot(gs[row, col],
                                 xscale = xscale,
                                 yscale = yscale)

            _format_axis(ax, param0_name, param1_name)

            for (k, x) in enumerate(parameters.parameter_sets.items()):
                n, p = x

                param0baseline = getattr(p, param0)
                param1baseline = getattr(p, param1)

                y = common.get_dPs(param0, param0baseline)
                x = common.get_dPs(param1, param1baseline)
                X, Y = numpy.meshgrid(x, y)

                cs = ax.contour(X, Y, rel_growth_rate[k, row, col],
                                contour_levels,
                                colors = [colors[k]],
                                alpha = alpha,
                                linestyles = 'solid')
                ax.clabel(cs, inline = 1, fmt = '%.4g', fontsize = 8,
                          colors = [colors[k]], alpha = alpha)

                ax.axvline(param1baseline, linestyle = 'dotted',
                           color = 'black', alpha = alpha)
                ax.axhline(param0baseline, linestyle = 'dotted',
                           color = 'black', alpha = alpha)

    handles = (lines.Line2D([], [], color = c, alpha = alpha)
               for c in seaborn.color_palette())
    labels = parameters.parameter_sets.keys()
    leg = fig.legend(handles, labels,
                     loc = 'upper right',
                     frameon = False,
                     fontsize = 'medium')

    fig.tight_layout()

    if save:
        common.savefig(fig)


if __name__ == '__main__':
    # rel_growth_rate = build()
    # numpy.save('sensitivity_2params.npy', rel_growth_rate)
    rel_growth_rate = numpy.load('sensitivity_2params.npy')
    plot(rel_growth_rate)
    pyplot.show()
