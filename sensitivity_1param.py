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

figsize = (8.5, 3)
figsize_epsilon = (8.5, 6)


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

            for (k, x) in enumerate(parameters.parameter_sets.items()):
                n, p = x

                param0baseline = getattr(p, param0)

                # We only need to draw these once.
                if k == 0:
                    ax.axvline(param0baseline, **common.baseline_style)

                dPs = common.get_dPs(param0, param0baseline)

                r0 = parallel(joblib.delayed(_run_one)(p, param0, dP)
                              for dP in dPs)
                r0 = numpy.asarray(r0)

                ymin = min(r0.min(), ymin)
                ymax = max(r0.max(), ymax)
                ax.plot(dPs, r0, label = n, alpha = common.alpha)

            ax.set_xlabel(param0_name, fontsize = 'x-small')
            if ax.is_first_col():
                ax.set_ylabel('Pathogen intrinsic growth rate (d$^{-1}$)',
                              fontsize = 'x-small')

            ax.yaxis.set_major_locator(ticker.LogLocator(subs = (1, )))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

    ymin = 10 ** numpy.floor(numpy.log10(ymin))
    ymax = 10 ** numpy.ceil(numpy.log10(ymax))
    ax.set_ylim(ymin, ymax)

    fig.tight_layout(rect = (0, 0.07, 1, 1))

    handles = (lines.Line2D([], [], color = c, alpha = common.alpha)
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
    linestyles = ('solid', 'dashed')
    fig, ax = pyplot.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize = 'small')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
    ax.xaxis.set_major_locator(ticker.LogLocator(subs = (1, 2, 5)))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ymin, ymax = (numpy.inf, - numpy.inf)
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        for (i, param0) in enumerate(('mu_f', 'mu_m')):
            colors = iter(seaborn.color_palette())
            for (n, p) in parameters.parameter_sets.items():
                param0baseline = getattr(p, param0)
                parammin = 1e-3
                mumax = p.mu * common.sensitivity_max_abs_mult_change
                if param0 == 'mu_f':
                    parammax = (mumax - (1 - p.phi) * p.mu_m) / p.phi
                else:
                    parammax = (mumax - p.phi * p.mu_f)  / (1 - p.phi)
                dPs = numpy.logspace(numpy.log10(parammin),
                                     numpy.log10(parammax),
                                     common.npoints)
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
                ax.plot(mu, r0, label = n, alpha = common.alpha,
                        color = next(colors), linestyle = linestyles[i])

            ax.set_xlabel('Death rate\n$\\mu$ (d$^{-1}$)',
                          fontsize = 'small')
            if ax.is_first_col():
                ax.set_ylabel('Pathogen intrinsic growth rate (d$^{-1}$)',
                              fontsize = 'small')

            mu0 = p.phi * p.mu_f + (1 - p.phi) * p.mu_m
            ax.axvline(mu0, linestyle = 'dotted', color = 'black',
                       alpha = common.alpha)

            ax.yaxis.set_major_locator(ticker.LogLocator(subs = (1, )))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

    ymin = 10 ** numpy.floor(numpy.log10(ymin))
    ymax = 10 ** numpy.ceil(numpy.log10(ymax))
    ax.set_ylim(ymin, ymax)

    fig.tight_layout(rect = (0, 0.07, 1, 1))

    handles = (lines.Line2D([], [], color = c, alpha = common.alpha)
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
    linestyles = ('solid', 'dashed', 'dotted')
    fig, ax = pyplot.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize = 'small')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
    ax.xaxis.set_major_locator(ticker.LogLocator(subs = (1, 2, 5)))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ymin, ymax = (numpy.inf, - numpy.inf)
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        for (i, param) in enumerate(common.sensitivity_parameters[0 : 3]):
            param0, param0_name = param

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
                ax.plot(R0, r0, label = n, alpha = common.alpha,
                        color = next(colors), linestyle = linestyles[i])

            ax.set_xlabel('Lifetime reproductive output\n$R_0$',
                          fontsize = 'small')
            if ax.is_first_col():
                ax.set_ylabel('Pathogen intrinsic growth rate (d$^{-1}$)',
                              fontsize = 'small')

            mu0 = p.phi * p.mu_f + (1 - p.phi) * p.mu_m
            R0_0 = p.rho * p.phi / mu0
            ax.axvline(R0_0, linestyle = 'dotted', color = 'black',
                       alpha = common.alpha)

            ax.yaxis.set_major_locator(ticker.LogLocator(subs = (1, )))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

    ymin = 10 ** numpy.floor(numpy.log10(ymin))
    ymax = 10 ** numpy.ceil(numpy.log10(ymax))
    ax.set_ylim(ymin, ymax)

    fig.tight_layout(rect = (0, 0.07, 1, 1))

    handles = (lines.Line2D([], [], color = c, alpha = common.alpha)
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


if __name__ == '__main__':
    main()
    # sensitivity_mu()
    # sensitivity_R0()
    pyplot.show()
