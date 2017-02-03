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
figsize_epsilon = (8.5, 6)
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
                              fontsize = 'x-small')

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


def sensitivity_mu():
    nparams = len(common.sensitivity_parameters)
    linestyles = ('solid', 'dashed')
    fig, ax = pyplot.subplots()
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        ymin, ymax = (numpy.inf, - numpy.inf)
        for (i, param) in enumerate(common.sensitivity_parameters[1 : 3]):
            param0, param0_name = param

            ax.set_xscale(common.get_scale(param0))
            ax.set_yscale('log')

            ax.tick_params(labelsize = 'small')

            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
            if ax.get_xscale() == 'linear':
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 4))
            elif ax.get_xscale() == 'log':
                ax.xaxis.set_major_locator(ticker.LogLocator(subs = (1, 2, 5)))
                ax.xaxis.set_minor_locator(ticker.NullLocator())

            colors = iter(seaborn.color_palette())
            for (n, p) in parameters.parameter_sets.items():
                param0baseline = getattr(p, param0)
                dPs = common.get_dPs(param0, param0baseline)
                r0 = parallel(joblib.delayed(_run_one)(p, param0, dP)
                              for dP in dPs)
                r0 = numpy.asarray(r0)
                ymin = min(r0.min(), ymin)
                ymax = max(r0.max(), ymax)
                if param0 == 'mu_f':
                    mu_f = dPs
                    mu_m = p.mu_m
                else:
                    mu_m = dPs
                    mu_f = p.mu_f
                mu = p.phi * mu_f + (1 - p.phi) * mu_m
                ax.plot(mu, r0, label = n, alpha = alpha,
                        color = next(colors), linestyle = linestyles[i])

            ax.set_xlabel('Death rate\n$\\mu$ (d$^{-1}$)',
                          fontsize = 'small')
            if ax.is_first_col():
                ax.set_ylabel('Pathogen intrinsic growth rate (d$^{-1}$)',
                              fontsize = 'small')

            mu0 = p.phi * p.mu_f + (1 - p.phi) * p.mu_m
            ax.axvline(mu0, linestyle = 'dotted', color = 'black',
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
        common.savefig(fig, append = '_mu')

    return fig


def sensitivity_R0():
    nparams = len(common.sensitivity_parameters)
    linestyles = ('solid', 'dashed', 'dotted')
    fig, ax = pyplot.subplots()
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        ymin, ymax = (numpy.inf, - numpy.inf)
        for (i, param) in enumerate(common.sensitivity_parameters[0 : 3]):
            param0, param0_name = param

            ax.set_xscale(common.get_scale(param0))
            ax.set_yscale('log')

            ax.tick_params(labelsize = 'small')

            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
            if ax.get_xscale() == 'linear':
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 4))
            elif ax.get_xscale() == 'log':
                ax.xaxis.set_major_locator(ticker.LogLocator(subs = (1, 2, 5)))
                ax.xaxis.set_minor_locator(ticker.NullLocator())

            colors = iter(seaborn.color_palette())
            for (n, p) in parameters.parameter_sets.items():
                param0baseline = getattr(p, param0)
                dPs = common.get_dPs(param0, param0baseline)
                r0 = parallel(joblib.delayed(_run_one)(p, param0, dP)
                              for dP in dPs)
                r0 = numpy.asarray(r0)
                ymin = min(r0.min(), ymin)
                ymax = max(r0.max(), ymax)
                if param0 == 'mu_f':
                    rho = p.rho
                    mu_f = dPs
                    mu_m = p.mu_m
                elif param0 == 'mu_m':
                    rho = p.rho
                    mu_f = p.mu_f
                    mu_m = dPs
                else:
                    rho = dPs
                    mu_f = p.mu_f
                    mu_m = p.mu_m
                mu = p.phi * mu_f + (1 - p.phi) * mu_m
                R0 = rho * p.phi / mu
                ax.plot(R0, r0, label = n, alpha = alpha,
                        color = next(colors), linestyle = linestyles[i])

            ax.set_xlabel('Lifetime reproductive output\n$R_0$',
                          fontsize = 'small')
            if ax.is_first_col():
                ax.set_ylabel('Pathogen intrinsic growth rate (d$^{-1}$)',
                              fontsize = 'small')

            mu0 = p.phi * p.mu_f + (1 - p.phi) * p.mu_m
            R0_0 = p.rho * p.phi / mu0
            ax.axvline(R0_0, linestyle = 'dotted', color = 'black',
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
        common.savefig(fig, append = '_R0')

    return fig


def sensitivity_epsilon():
    # Get long name.
    for p_, n in common.sensitivity_parameters:
        if p_ == 'epsilon':
            label = n
            break
    else:
        label = 'epsilon'

    xscale = 'linear'
    yscale = 'linear'

    fig, ax = pyplot.subplots(figsize = figsize_epsilon,
                              subplot_kw = dict(xscale = xscale,
                                                yscale = yscale))

    dPs = numpy.linspace(0, 10, 1001)
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        for (n, p) in parameters.parameter_sets.items():
            epsilon_baseline = p.epsilon
            r0 = parallel(joblib.delayed(_run_one)(p, 'epsilon', dP)
                          for dP in dPs)
            l = ax.plot(dPs, r0, label = n, alpha = alpha)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 4))
    ax.set_xlabel(label, fontsize = 'small')
    ax.set_ylabel('Pathogen intrinsic growth rate (d${-1}$)',
                  fontsize = 'small')
    ax.tick_params(labelsize = 'small')

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

    ax.axvline(epsilon_baseline, linestyle = 'dotted', color = 'black',
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
        common.savefig(fig, append = '_epsilon')

    return fig


if __name__ == '__main__':
    main()
    sensitivity_epsilon()
    sensitivity_mu()
    sensitivity_R0()
    pyplot.show()
