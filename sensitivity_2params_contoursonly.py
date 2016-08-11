#!/usr/bin/python3

from matplotlib import gridspec
from matplotlib import lines
from matplotlib import pyplot
from matplotlib import ticker
import numpy

import common
import parameters


save = False

t = 150
figsize = (8.5, 8)
common.seaborn.set_palette('Dark2')
alpha = 0.7


def build():
    nparamsets = len(parameters.parameter_sets)
    nparams = len(common.sensitivity_parameters)
    rel_growth_rate = numpy.ones((nparamsets,
                                  nparams - 1, nparams - 1,
                                  common.npoints,
                                  common.npoints))
    for (k, p) in enumerate(parameters.parameter_sets.values()):
        r0baseline = p.QSSA.r0(t)
        for i in range(nparams - 1):
            for j in range(i + 1):
                param0, param0_name = common.sensitivity_parameters[i + 1]
                param1, param1_name = common.sensitivity_parameters[j]
                param0baseline = getattr(p, param0)
                param1baseline = getattr(p, param1)
                dPs0 = common.get_dPs(param0, param0baseline)
                for (m, dP0) in enumerate(dPs0):
                    setattr(p, param0, dP0)
                    dPs1 = common.get_dPs(param1, param1baseline)
                    for (n, dP1) in enumerate(dPs1):
                        setattr(p, param1, dP1)
                        rel_growth_rate[k, i, j, m, n] = (p.QSSA.r0(t)
                                                          / r0baseline)
                setattr(p, param0, param0baseline)
                setattr(p, param1, param1baseline)
    return rel_growth_rate


def plot(rel_growth_rate):
    # Contour levels every log10.
    log_rgr = numpy.log10(rel_growth_rate)
    log_rgr_absmax = numpy.max(numpy.abs(log_rgr))
    log_rgr_levels = numpy.linspace(- numpy.ceil(log_rgr_absmax),
                                    numpy.ceil(log_rgr_absmax),
                                    2 * numpy.ceil(log_rgr_absmax) + 1)
    contour_levels = 10 ** log_rgr_levels

    colors = common.seaborn.color_palette()

    nparams = len(common.sensitivity_parameters)
    nrows = nparams - 1
    ncols = nparams - 1
    gs = gridspec.GridSpec(nrows, ncols)
    fig = pyplot.figure(figsize = figsize)
    for row in range(nparams - 1):
        for col in range(row + 1):
            param0, param0_name = common.sensitivity_parameters[row + 1]
            param1, param1_name = common.sensitivity_parameters[col]

            yscale = common.get_scale(param0)
            xscale = common.get_scale(param1)

            ax = fig.add_subplot(gs[row, col],
                                 xscale = xscale,
                                 yscale = yscale)

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

                if ax.is_last_row():
                    ax.set_xlabel(param1_name, fontsize = 9)
                else:
                    for l in ax.xaxis.get_ticklabels():
                        l.set_visible(False)
                    ax.xaxis.offsetText.set_visible(False)
                if ax.is_first_col():
                    ax.set_ylabel(param0_name, fontsize = 9)
                else:
                    for l in ax.yaxis.get_ticklabels():
                        l.set_visible(False)
                    ax.yaxis.offsetText.set_visible(False)

                for axis in (ax.xaxis, ax.yaxis):
                    scale = axis.get_scale()
                    if scale == 'linear':
                        axis.set_major_locator(ticker.MaxNLocator(nbins = 4))
                    elif scale == 'log':
                        axis.set_major_locator(ticker.LogLocator(
                            subs = [1, 2, 5]))
                    axis.set_major_formatter(ticker.StrMethodFormatter(
                        '{x:g}'))
                    ax.tick_params(labelsize = 9)

    fig.tight_layout()

    handles = (lines.Line2D([], [], color = c, alpha = alpha)
               for c in common.seaborn.color_palette())
    labels = parameters.parameter_sets.keys()
    leg = fig.legend(handles, labels,
                     loc = 'upper right',
                     frameon = False,
                     fontsize = 'medium')
    if save:
        common.savefig(fig)


if __name__ == '__main__':
    rel_growth_rate = build()
    plot(rel_growth_rate)
    pyplot.show()
