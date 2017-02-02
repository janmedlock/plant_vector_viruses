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

figsize = (8.5, 3)
figsize_fV = (8.5, 6)
seaborn.set_palette('Dark2')
alpha = 0.7


def _run_one(p, param0, dP):
    p = copy.copy(p)
    setattr(p, param0, dP)
    return growth_rates.get_growth_rate(p)


def main():
    nparams = len(common.sensitivity_parameters)
    fig, axes = pyplot.subplots(1, nparams,
                                sharey = True,
                                figsize = figsize)
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        ymin, ymax = (numpy.inf, - numpy.inf)
        for (i, param) in enumerate(common.sensitivity_parameters):
            param0, param0_name = param

            ax = axes[i]
            # ax.autoscale(tight = True)  # Bug!
            ax.set_xscale(common.get_scale(param0))
            ax.set_yscale('log')

            ax.tick_params(labelsize = 'x-small')

            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
            if ax.get_xscale() == 'linear':
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 4))
            elif ax.get_xscale() == 'log':
                ax.xaxis.set_major_locator(ticker.LogLocator(subs = (1, 2, 5)))

            for (n, p) in parameters.parameter_sets.items():
                param0baseline = getattr(p, param0)
                dPs = common.get_dPs(param0, param0baseline)
                r0 = parallel(joblib.delayed(_run_one)(p, param0, dP)
                              for dP in dPs)
                r0 = numpy.asarray(r0)
                ymin = min(r0.min(), ymin)
                ymax = max(r0.max(), ymax)
                ax.plot(dPs, r0, label = n, alpha = alpha)

            ax.set_xlabel(param0_name, fontsize = 'x-small')
            if ax.is_first_col():
                ax.set_ylabel('Pathogen intrinsic growth rate (d$^{-1}$)',
                              fontsize = 'small')

            ax.axvline(param0baseline, linestyle = 'dotted', color = 'black',
                       alpha = alpha)

            ax.yaxis.set_major_locator(ticker.LogLocator(subs = (1, )))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

    ymin = 10 ** numpy.floor(numpy.log10(ymin))
    ymax = 10 ** numpy.ceil(numpy.log10(ymax))
    ax.set_ylim(ymin, ymax)

    fig.tight_layout(rect = (0, 0.07, 1, 1))

    handles = (lines.Line2D([], [], color = c, alpha = alpha)
               for c in seaborn.color_palette())
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


def sensitivity_fV_only():
    # Get long name.
    for p_, n in common.sensitivity_parameters:
        if p_ == 'fV':
            label = n
            break
    else:
        label = 'fV'

    xscale = 'linear'
    yscale = 'linear'

    fig, ax = pyplot.subplots(figsize = figsize_fV,
                              subplot_kw = dict(xscale = xscale,
                                                yscale = yscale))

    dPs = numpy.linspace(0, 10, 1001)
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        for (n, p) in parameters.parameter_sets.items():
            fV_baseline = p.fV
            r0 = parallel(joblib.delayed(_run_one)(p, 'fV', dP)
                          for dP in dPs)
            l = ax.plot(dPs, r0, label = n, alpha = alpha)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 4))
    ax.set_xlabel(label, fontsize = 'x-small')
    ax.set_ylabel('Pathogen intrinsic growth rate (d${-1}$)',
                  fontsize = 'small')
    ax.tick_params(labelsize = 'x-small')

    if xscale == 'linear':
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 6))
    elif xscale == 'log':
        ax.xaxis.set_major_locator(ticker.LogLocator(subs = [1, 2, 5]))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

    if yscale == 'log':
        ax.yaxis.set_major_locator(ticker.LogLocator(subs = [1, 2, 5]))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

    ax.axvline(fV_baseline, linestyle = 'dotted', color = 'black',
               alpha = alpha)

    handles = (lines.Line2D([], [], color = c, alpha = alpha)
               for c in seaborn.color_palette())
    labels = parameters.parameter_sets.keys()
    fig.legend(handles, labels,
               loc = 'lower center',
               ncol = len(parameters.parameter_sets),
               columnspacing = 10,
               frameon = False,
               fontsize = 'small',
               numpoints = 1)

    fig.tight_layout(rect = (0, 0.04, 1, 1))

    if save:
        common.savefig(fig, append = '_fV')

    return fig


if __name__ == '__main__':
    main()
    sensitivity_fV_only()
    pyplot.show()
