#!/usr/bin/python3

import copy
import itertools
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

figsize = (8.5, 4)


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
        for (k, paramset) in enumerate(parameters.parameter_sets.items()):
            p_name, p = paramset
            print('Running parameter set {}.'.format(p_name))
            ij = 0
            for i in range(nparams - 1):
                param0, param0_name = common.sensitivity_parameters[i]
                param0baseline = getattr(p, param0)
                dPs0 = common.get_dPs(param0, param0baseline)
                for j in range(i + 1, nparams):
                    param1, param1_name = common.sensitivity_parameters[j]
                    param1baseline = getattr(p, param1)
                    dPs1 = common.get_dPs(param1, param1baseline)
                    print('\tRunning {} and {}.'.format(param0, param1))
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
    return '{}{}'.format(sign, m.group(1))


def _get_xlabel(param_names, m):
    m = numpy.asarray(m)
    # Label for x > 0
    sp = ','.join(itertools.starmap(_get_sym, zip(param_names, m)))
    # Label for x < 0
    sn = ','.join(itertools.starmap(_get_sym, zip(param_names, -m)))
    return '${} \\quad {}$'.format(sn, sp)


def plot(r0):
    arrowpos = - 0.18
    nparams = len(common.sensitivity_parameters)
    npairs = nparams * (nparams - 1) // 2
    fig, axes = pyplot.subplots(2, npairs,
                                sharey = True,
                                figsize = figsize,
                                squeeze = False)
    x0 = (r0.shape[-1] - 1) / 2
    ij = 0
    for i in range(nparams - 1):
        for j in range(i + 1, nparams):
            param0, param0_name = common.sensitivity_parameters[i]
            param1, param1_name = common.sensitivity_parameters[j]
            for l in range(axes.shape[0]):
                ax = axes[l, ij]
                if l == 0:
                    m = (1, 1)
                    d = 0
                elif l == 1:
                    m = (1, -1)
                    d = 1
                ax.axvline(x0, ymin = arrowpos - 0.025,
                           clip_on = False,
                           **common.baseline_style)
                for (k, n) in enumerate(parameters.parameter_sets.keys()):
                    ax.plot(r0[k, ij, d, : : m[0]], label = n,
                            alpha = common.alpha)
                param_names = (param0_name, param1_name)
                xlabel = _get_xlabel(param_names, m)
                ax.set_xlabel(xlabel, fontsize = 'x-small')
                # ax.autoscale(tight = True)  # Bug!
                ax.set_yscale('log')
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                for i in (-1, +1):
                    ax.annotate('',
                                xy = (0.5 + 0.45 * i, arrowpos),
                                xytext = (0.5 + 0.05 * i, arrowpos),
                                xycoords = 'axes fraction',
                                annotation_clip = False,
                                arrowprops = dict(arrowstyle = '->',
                                                  linewidth = 0.6))
            ij += 1

    fig.tight_layout(rect = (0.02, 0.075, 1, 1), h_pad = 1.4)

    fig.text(0.01, 0.55,
             'Pathogen intrinsic growth rate (d$^{-1}$)',
             ha = 'left',
             va = 'center',
             rotation = 'vertical',
             fontsize = 'x-small')

    handles = [lines.Line2D([], [], color = c, alpha = common.alpha)
               for c in seaborn.color_palette()]
    labels = list(parameters.parameter_sets.keys())
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
    r0 = common.load_or_build_data(build)
    plot(r0)
    pyplot.show()
