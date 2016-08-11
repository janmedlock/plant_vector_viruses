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


def plot_sensitivity(p):
    nparams = len(common.sensitivity_parameters)
    r00 = p.QSSA.r0(t)
    r0 = numpy.ones((nparams - 1, nparams - 1,
                     len(common.sensitivity_dPs),
                     len(common.sensitivity_dPs)))
    for i in range(nparams - 1):
        for j in range(i + 1):
            param0, param0_name = common.sensitivity_parameters[i + 1]
            param1, param1_name = common.sensitivity_parameters[j]
            param00 = getattr(p, param0)
            param10 = getattr(p, param1)
            for (m, dP0) in enumerate(common.sensitivity_dPs):
                setattr(p, param0, param00 * dP0)
                for (n, dP1) in enumerate(common.sensitivity_dPs):
                    setattr(p, param1, param10 * dP1)
                    r0[i, j, m, n] = p.QSSA.r0(t)
            setattr(p, param0, param00)
            setattr(p, param1, param10)
    Z = r0 / r00

    norm = colors.LogNorm()
    norm(Z)  # Set limits.
    # Put white at 1.
    cmap = shifted_cmap(cmap_base, norm(1))

    # Contour levels every log10.
    T = numpy.log10(Z)
    Tabsmax = numpy.max(numpy.abs(T))
    TV = numpy.linspace(- numpy.ceil(Tabsmax),
                        numpy.ceil(Tabsmax),
                        2 * numpy.ceil(Tabsmax) + 1)
    V = 10 ** TV

    # Colorbar in a small extra column.
    nrows = nparams - 1
    ncols = nparams
    gs = gridspec.GridSpec(nrows, ncols,
                           width_ratios = ([1] * (ncols - 1)
                                           + [colorbar_width_ratio]))
    fig = pyplot.figure(figsize = figsize)
    for row in range(nparams - 1):
        for col in range(row + 1):
            param0, param0_name = common.sensitivity_parameters[row + 1]
            param1, param1_name = common.sensitivity_parameters[col]

            ax = fig.add_subplot(gs[row, col],
                                 xscale = 'log',
                                 yscale = 'log',
                                 axisbg = 'none')

            param00 = getattr(p, param0)
            param10 = getattr(p, param1)

            y = param00 * common.sensitivity_dPs
            x = param10 * common.sensitivity_dPs
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

            pc = ax.pcolormesh(X, Y, Z[row, col],
                               cmap = cmap, norm = norm,
                               shading = 'gouraud')

            cs = ax.contour(X, Y, Z[row, col], V,
                            colors = 'black',
                            linewidths = 1,
                            linestyles = 'solid')
            ax.clabel(cs, inline = 1, fmt = '%.4g', fontsize = 8)

            ax.axvline(param10, linestyle = 'dotted', color = 'black',
                       alpha = 0.5)
            ax.axhline(param00, linestyle = 'dotted', color = 'black',
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


def main():
    for (n, p) in parameters.parameter_sets.items():
        fig = plot_sensitivity(p)
        common.savefig(fig,
                       append = '_{}'.format(n.lower()))


if __name__ == '__main__':
    main()
    # pyplot.show()

    # Make scale the same.
