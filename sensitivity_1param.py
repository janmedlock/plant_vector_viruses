#!/usr/bin/python3

from matplotlib import gridspec
from matplotlib import lines
from matplotlib import pyplot
from matplotlib import ticker
import numpy

import common
import growth_rates
import parameters


save = True

figsize = (8.5, 3)
common.seaborn.set_palette('Dark2')
alpha = 0.7


def main():
    nparams = len(common.sensitivity_parameters)
    fig = pyplot.figure(figsize = figsize)
    gs = gridspec.GridSpec(1, nparams)
    sharey = None
    ymin = ymax = None
    for (i, param) in enumerate(common.sensitivity_parameters):
        param0, param0_name = param

        xscale = common.get_scale(param0)
        ax = fig.add_subplot(gs[0, i],
                             sharey = sharey,
                             xscale = xscale, yscale = 'log')
        if sharey is None:
            sharey = ax

        for (n, p) in parameters.parameter_sets.items():
            r0baseline = growth_rates.get_growth_rate(p)

            param0baseline = getattr(p, param0)

            dPs = common.get_dPs(param0, param0baseline)
            r0 = numpy.empty(len(dPs))
            for (j, dP0) in enumerate(dPs):
                setattr(p, param0, dP0)
                r0[j] = growth_rates.get_growth_rate(p) / r0baseline
            setattr(p, param0, param0baseline)

            l = ax.plot(dPs, r0, label = n, alpha = alpha)

            if ymax is None:
                ymax = max(r0)
            else:
                ymax = max(max(r0), ymax)
            if ymin is None:
                ymin = min(r0)
            else:
                ymin = min(min(r0), ymin)

        ax.set_xlim(min(dPs), max(dPs))

        ax.set_xlabel(param0_name, fontsize = 'x-small')
        if ax.is_first_col():
            ax.set_ylabel('Relative infection\ngrowth rate',
                          fontsize = 'small')
        else:
            for l in ax.yaxis.get_ticklabels():
                l.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

        ax.tick_params(labelsize = 'x-small')

        if xscale == 'linear':
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 4))
        elif xscale == 'log':
            ax.xaxis.set_major_locator(ticker.LogLocator(subs = [1, 2, 5]))
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

        ax.axvline(param0baseline, linestyle = 'dotted', color = 'black',
                   alpha = alpha)

    # ymin = 10 ** numpy.floor(numpy.log10(ymin))
    # ymax = 10 ** numpy.ceil(numpy.log10(ymax))
    ax.set_ylim(ymin, ymax)

    fig.tight_layout(rect = (0, 0.07, 1, 1))
    
    handles = (lines.Line2D([], [], color = c, alpha = alpha)
               for c in common.seaborn.color_palette())
    labels = parameters.parameter_sets.keys()
    leg = fig.legend(handles, labels,
                     loc = 'lower center',
                     ncol = len(labels),
                     columnspacing = 10,
                     frameon = False,
                     fontsize = 'small',
                     numpoints = 1)

    if save:
        common.savefig(fig)

    return fig


if __name__ == '__main__':
    main()
    pyplot.show()
