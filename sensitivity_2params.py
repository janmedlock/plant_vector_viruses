#!/usr/bin/python3

import itertools

from matplotlib import cm
from matplotlib import colorbar
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot
from matplotlib import ticker
import numpy

import common
import parameters


t = (0, 50, 100, 150)
cmap_base = 'RdBu'
figsize = (8, 8)


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
    pairs = list(itertools.combinations(common.sensitivity_parameters,
                                        2))

    fig = pyplot.figure(figsize = figsize)
    gs = gridspec.GridSpec(len(t), len(pairs) + 1,
                           width_ratios = (1, ) * len(pairs) + (0.3, ))

    for (i, t_) in enumerate(t):
        r00 = p.QSSA.r0(t_)
        r0 = numpy.empty((len(pairs),
                          len(common.sensitivity_dPs),
                          len(common.sensitivity_dPs)))
        for (j, p0p1) in enumerate(pairs):
            p0, p1 = p0p1
            param0, param0_name = p0
            param1, param1_name = p1
            param00 = getattr(p, param0)
            param10 = getattr(p, param1)
            for (m, dP0) in enumerate(common.sensitivity_dPs):
                setattr(p, param0, param00 * dP0)
                for (n, dP1) in enumerate(common.sensitivity_dPs):
                    setattr(p, param1, param10 * dP1)
                    r0[j, n, m] = p.QSSA.r0(t_)
            setattr(p, param0, param00)
            setattr(p, param1, param10)


        Z = r0

        norm = colors.LogNorm()
        norm(Z)  # Set limits.
        # Put white at r00.
        cmap = shifted_cmap(cmap_base, norm(r00))

        # Contour levels every log10.
        T = numpy.log10(Z / r00)
        Tabsmax = numpy.max(numpy.abs(T))
        TV = numpy.linspace(- numpy.ceil(Tabsmax),
                            numpy.ceil(Tabsmax),
                            2 * numpy.ceil(Tabsmax) + 1)
        V = r00 * 10 ** TV

        for (j, p0p1) in enumerate(pairs):
            p0, p1 = p0p1
            param0, param0_name = p0
            param1, param1_name = p1
            param00 = getattr(p, param0)
            param10 = getattr(p, param1)

            ax = fig.add_subplot(gs[i, j],
                                 xscale = 'log',
                                 yscale = 'log')

            x = param00 * common.sensitivity_dPs
            y = param10 * common.sensitivity_dPs
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

            pc = ax.pcolormesh(X, Y, Z[j], cmap = cmap, norm = norm,
                               shading = 'gouraud')

            cs = ax.contour(X, Y, Z[j], V,
                            colors = 'black',
                            linewidths = 1,
                            linestyles = 'solid')
            ax.clabel(cs, inline = 1, fmt = '%.4g', fontsize = 8)

            ax.set_xlabel(param0_name, fontsize = 10)
            ax.set_ylabel(param1_name, fontsize = 10)
            ax.tick_params(labelsize = 7)
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(min(y), max(y))

            xlim = numpy.log(ax.get_xlim())
            ylim = numpy.log(ax.get_ylim())
            aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            ax.set_aspect(aspect, 'box-forced')

            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(ticker.LogLocator(subs = [1, 2, 5]))
                axis.set_major_formatter(ticker.ScalarFormatter())

            if j == len(pairs) // 2:
                ax.set_title('At {} d'.format(t_))

            ax.axvline(param00, linestyle = 'dotted', color = 'black',
                       alpha = 0.5)
            ax.axhline(param10, linestyle = 'dotted', color = 'black',
                       alpha = 0.5)

        ax = fig.add_subplot(gs[i, -1])
        cbar = colorbar.Colorbar(ax, pc,
                                 label = 'Infection growth rate (d$^{-1}$)',
                                 orientation = 'vertical')
        cbar.ax.axhline(norm(r00), linestyle = 'dotted', color = 'black',
                        alpha = 0.5)
        cbar.ax.tick_params(labelsize = 10)

    fig.tight_layout()

    return fig


def main():
    for (n, p) in parameters.parameter_sets.items():
        fig = plot_sensitivity(p)
        common.savefig(fig,
                       append = '_{}'.format(n.lower()))


if __name__ == '__main__':
    main()
    pyplot.show()
