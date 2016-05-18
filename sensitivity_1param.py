#!/usr/bin/python3

from matplotlib import colorbar
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot
from matplotlib import ticker
import numpy

import common
import parameters


cmap = 'viridis'
figsize = (8, 4)


def plot_sensitivity(p):
    nparams = len(common.sensitivity_parameters)
    r00 = p.QSSA.r0(common.t)
    r0 = numpy.empty((nparams,
                      len(common.sensitivity_dPs),
                      len(common.t)))
    for (i, param) in enumerate(common.sensitivity_parameters):
        param0, param0_name = param
        param00 = getattr(p, param0)
        for (j, dP0) in enumerate(common.sensitivity_dPs):
            setattr(p, param0, param00 * dP0)
            r0[i, j] = p.QSSA.r0(common.t)
            setattr(p, param0, param00)

    fig = pyplot.figure(figsize = figsize)
    gs = gridspec.GridSpec(1, nparams + 1,
                           width_ratios = (1, ) * nparams + (0.3, ))
    norm = colors.LogNorm()
    norm(r0)  # Set limits.
    for (i, param) in enumerate(common.sensitivity_parameters):
        param0, param0_name = param
        param00 = getattr(p, param0)

        x = common.t
        y = param00 * common.sensitivity_dPs
        X, Y = numpy.meshgrid(x, y)
        Z = r0[i]

        # Contour levels every log10.
        T = numpy.log10(Z)
        Tabsmax = numpy.max(numpy.abs(T))
        TV = numpy.linspace(- numpy.ceil(Tabsmax),
                            numpy.ceil(Tabsmax),
                            2 * numpy.ceil(Tabsmax) + 1)
        V = 10 ** TV

        ax = fig.add_subplot(gs[0, i], yscale = 'log')
        pc = ax.pcolormesh(X, Y, Z, cmap = cmap, norm = norm,
                           shading = 'gouraud')
        cs = ax.contour(X, Y, Z, V,
                        colors = 'black',
                        linewidths = 1,
                        linestyles = 'solid')
        ax.clabel(cs, inline = 1, fmt = '%.4g', fontsize = 8)

        ax.set_xlabel('Time (d)', fontsize = 12)
        ax.set_ylabel(param0_name, fontsize = 12)
        ax.tick_params(labelsize = 7)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))

        ax.yaxis.set_major_locator(ticker.LogLocator(subs = [1, 2, 5]))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        ax.axhline(param00, linestyle = 'dotted', color = 'black',
                   alpha = 0.5)

        common.style_axis(ax)

    ax = fig.add_subplot(gs[0, -1])
    cbar = colorbar.Colorbar(ax, pc,
                             label = 'Infection growth rate (d$^{-1}$)',
                             orientation = 'vertical')
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
