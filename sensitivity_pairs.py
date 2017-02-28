#!/usr/bin/python3

import copy
import re

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

figsize = (8.5, 8)


def _run_one(p, param0, param1, dP0, dP1):
    p = copy.copy(p)
    setattr(p, param0, dP0)
    setattr(p, param1, dP1)
    return growth_rates.get_growth_rate(p)


def build():
    nparamsets = len(parameters.parameter_sets)
    nparams = len(common.sensitivity_parameters)
    npairs = nparams * (nparams - 1) // 2
    ndirs = 2
    r0 = numpy.ones((nparamsets, npairs, ndirs, common.npoints))
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        for (k, p) in enumerate(parameters.parameter_sets.values()):
            ij = 0
            for i in range(nparams - 1):
                for j in range(i + 1, nparams):
                    param0, param0_name = common.sensitivity_parameters[i]
                    param1, param1_name = common.sensitivity_parameters[j]
                    print('Running {} and {}.'.format(param0, param1))
                    param0baseline = getattr(p, param0)
                    param1baseline = getattr(p, param1)
                    dPs0 = common.get_dPs(param0, param0baseline)
                    dPs1 = common.get_dPs(param1, param1baseline)
                    for l in range(2):
                        if l == 1:
                            dPs1 = dPs1[ : : -1]
                        r0[k, ij, l] = parallel(
                            joblib.delayed(_run_one)(p,
                                                     param0,
                                                     param1,
                                                     dP0,
                                                     dP1)
                            for (dP0, dP1) in zip(dPs0, dPs1))
                    ij += 1
    return r0



_sym_regex = re.compile(r'\$([^$]+)\$')
def _get_sym(pn, s):
    sign = '+' if s > 0 else '-'
    m = _sym_regex.search(pn)
    return '{} {}'.format(sign, m.group(1))


def plot(r0):
    nparams = len(common.sensitivity_parameters)
    npairs = nparams * (nparams - 1) // 2
    with seaborn.axes_style('dark'):
        fig, axes = pyplot.subplots(4, npairs,
                                    sharey = True,
                                    figsize = figsize,
                                    squeeze = False)
    ij = 0
    for i in range(nparams - 1):
        for j in range(i + 1, nparams):
            param0, param0_name = common.sensitivity_parameters[i]
            param1, param1_name = common.sensitivity_parameters[j]
            for l in range(4):
                ax = axes[l, ij]
                if l == 0:
                    m = (1, 1)
                    d = 0
                elif l == 1:
                    m = (-1, -1)
                    d = 0
                elif l == 2:
                    m = (1, -1)
                    d = 1
                else:
                    m = (-1, 1)
                    d = 1
                for (k, n) in enumerate(parameters.parameter_sets.keys()):
                    ax.plot(r0[k, ij, d, : : m[0]], label = n,
                            alpha = common.alpha)
                    # We only need to draw these once.
                    if k == 0:
                        x0 = (r0.shape[-1] - 1) / 2
                        ax.axvline(x0, **common.baseline_style)
                s = '${}, {}$'.format(_get_sym(param0_name, m[0]),
                                      _get_sym(param1_name, m[1]))
                ax.set_xlabel(s, fontsize = 'x-small')
                # ax.autoscale(tight = True)  # Bug!
                ax.set_yscale('log')
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
            ij += 1

    fig.tight_layout(rect = (0.03, 0.03, 1, 1))

    fig.text(0.01, 0.55,
             'Pathogen intrinsic growth rate (d$^{-1}$)',
             ha = 'left',
             va = 'center',
             rotation = 'vertical',
             fontsize = 'x-small')

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


if __name__ == '__main__':
    # r0 = build()
    # numpy.save('sensitivity_pairs.npy', r0)
    r0 = numpy.load('sensitivity_pairs.npy')
    plot(r0)
    pyplot.show()
