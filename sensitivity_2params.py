#!/usr/bin/python3

from matplotlib import cm
from matplotlib import colorbar
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot
from matplotlib import ticker
import numpy

import common
import parameters

pyplot.rcParams['mathtext.fontset'] = "stix"


save = True


t = 150
cmap_base = 'RdBu'
figsize = (8.5, 7.25)
colorbar_width_ratio = 0.3


def shifted_cmap(cmap, midpoint=0.5):
    '''
    Function to offset the "center" of a colormap.

    cmap:      The matplotlib colormap to be altered
    midpoint:  The new center of the colormap.
               Defaults to 0.5 (no shift).
               Must be between 0 and 1.
               For a midpoint at vmid, set midpoint to
                 (vmid - vmin) / (vmax - vmin).
    '''
    assert (0 <= midpoint <= 1), 'midpoint must be between 0 and 1.'

    cmap_ = cm.get_cmap(cmap)

    def trans(x):
        if x < 0.5:
            return 2 * midpoint * x
        else:
            return 1 - 2 * (1 - midpoint) * (1 - x)

    cdict = {}
    for (k, v) in cmap_._segmentdata.items():
        cdict[k] = [(trans(x), y1, y2) for (x, y1, y2) in v]
    name = '{}_shifted'.format(cmap_.name)
    return colors.LinearSegmentedColormap(name, cdict)


def get_scale(param):
    if param == 'phiV':
        return 'linear'
    else:
        return 'log'


def get_dPs(param, value_baseline):
    scale = get_scale(param)
    if scale == 'linear':
        eps = 0.1
        return numpy.linspace(eps, 1 - eps,
                              len(common.sensitivity_dPs))
    elif scale == 'log':
        return value_baseline * common.sensitivity_dPs
    else:
        raise NotImplementedError('scale = {}'.format(scale))


def plot_sensitivity(p, rel_growth_rate, contour_levels, cmap, norm):
    nparams = len(common.sensitivity_parameters)
    nrows = nparams - 1
    ncols = nparams - 1 + 1  # Colorbar in a small extra column.
    gs = gridspec.GridSpec(nrows, ncols,
                           width_ratios = ([1] * (ncols - 1)
                                           + [colorbar_width_ratio]))
    fig = pyplot.figure(figsize = figsize)
    for row in range(nparams - 1):
        for col in range(row + 1):
            param0, param0_name = common.sensitivity_parameters[row + 1]
            param1, param1_name = common.sensitivity_parameters[col]

            yscale = get_scale(param0)
            xscale = get_scale(param1)
            ax = fig.add_subplot(gs[row, col],
                                 xscale = xscale,
                                 yscale = yscale,
                                 axisbg = 'none')

            param0baseline = getattr(p, param0)
            param1baseline = getattr(p, param1)

            y = get_dPs(param0, param0baseline)
            x = get_dPs(param1, param1baseline)
            X, Y = numpy.meshgrid(x, y)

            # For pcolor: Z[j, k] is
            # in the middle of the rectangle with corners
            # (X_[j, k], Y_[j, k]) and
            # (X_[j + 1, k + 1], Y_[j + 1, k + 1])).
            # x_ = numpy.hstack((x[0] - (x[1] - x[0]) / 2,
            #                   (x[1 : ] + x[: -1]) / 2,
            #                   x[-1] + (x[-1] - x[-2]) / 2))
            # y_ = numpy.hstack((y[0] - (y[1] - y[0]) / 2,
            #                   (y[1 : ] + y[: -1]) / 2,
            #                   y[-1] + (y[-1] - y[-2]) / 2))
            # X_, Y_ = numpy.meshgrid(x_, y_)

            pc = ax.pcolormesh(X, Y, rel_growth_rate[row, col],
                               cmap = cmap, norm = norm,
                               shading = 'gouraud')

            cs = ax.contour(X, Y, rel_growth_rate[row, col],
                            contour_levels,
                            colors = 'black',
                            linewidths = 1,
                            linestyles = 'solid')
            ax.clabel(cs, inline = 1, fmt = '%.4g', fontsize = 8)

            ax.axvline(param1baseline, linestyle = 'dotted', color = 'black',
                       alpha = 0.5)
            ax.axhline(param0baseline, linestyle = 'dotted', color = 'black',
                       alpha = 0.5)

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
                    axis.set_major_locator(ticker.MaxNLocator(nbins = 5))
                elif scale == 'log':
                    axis.set_major_locator(ticker.LogLocator(subs = [1, 2, 5]))
                axis.set_major_formatter(ticker.ScalarFormatter())
                ax.tick_params(labelsize = 9)

            # xlim = numpy.log(ax.get_xlim())
            # ylim = numpy.log(ax.get_ylim())
            # aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            # ax.set_aspect(aspect, 'box-forced')

    ax = fig.add_subplot(gs[:, -1])
    cbar = colorbar.Colorbar(ax, pc,
                             label = 'Relative infection growth rate',
                             orientation = 'vertical')
    cbar.ax.tick_params(labelsize = 10)
    # cbar.ax.axhline(norm(1), linestyle = 'dotted', color = 'black',
    #                 alpha = 0.5)

    fig.tight_layout()

    return fig


def build():
    nparamsets = len(parameters.parameter_sets)
    nparams = len(common.sensitivity_parameters)
    rel_growth_rate = numpy.ones((nparamsets,
                                  nparams - 1, nparams - 1,
                                  len(common.sensitivity_dPs),
                                  len(common.sensitivity_dPs)))
    for (k, p) in enumerate(parameters.parameter_sets.values()):
        r0baseline = p.QSSA.r0(t)
        for i in range(nparams - 1):
            for j in range(i + 1):
                param0, param0_name = common.sensitivity_parameters[i + 1]
                param1, param1_name = common.sensitivity_parameters[j]
                param0baseline = getattr(p, param0)
                param1baseline = getattr(p, param1)
                dPs0 = get_dPs(param0, param0baseline)
                for (m, dP0) in enumerate(dPs0):
                    setattr(p, param0, dP0)
                    dPs1 = get_dPs(param1, param1baseline)
                    for (n, dP1) in enumerate(dPs1):
                        setattr(p, param1, dP1)
                        rel_growth_rate[k, i, j, m, n] = (p.QSSA.r0(t)
                                                          / r0baseline)
                setattr(p, param0, param0baseline)
                setattr(p, param1, param1baseline)
    return rel_growth_rate


def plot(rel_growth_rate):
    norm = colors.LogNorm()
    norm(rel_growth_rate)  # Set limits.
    # Put white at 1.
    cmap = shifted_cmap(cmap_base, norm(1))

    # Contour levels every log10.
    log_rgr = numpy.log10(rel_growth_rate)
    log_rgr_absmax = numpy.max(numpy.abs(log_rgr))
    log_rgr_levels = numpy.linspace(- numpy.ceil(log_rgr_absmax),
                                    numpy.ceil(log_rgr_absmax),
                                    2 * numpy.ceil(log_rgr_absmax) + 1)
    contour_levels = 10 ** log_rgr_levels

    for (k, x) in enumerate(parameters.parameter_sets.items()):
        n, p = x
        fig = plot_sensitivity(p, rel_growth_rate[k],
                               contour_levels, cmap, norm)
        if save:
            common.savefig(fig,
                           append = '_{}'.format(n.lower()))


if __name__ == '__main__':
    rel_growth_rate = build()
    plot(rel_growth_rate)
    pyplot.show()
