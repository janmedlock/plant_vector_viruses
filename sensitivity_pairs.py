#!/usr/bin/python3

import copy
import itertools

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


def convert_to_dict(r0_):
    nparams = len(common.sensitivity_parameters)
    r0 = {n: {} for n in parameters.parameter_sets.keys()}
    x0 = int((r0_.shape[-1] - 1) / 2)
    ij = 0
    for i in range(nparams - 1):
        for j in range(i + 1, nparams):
            param0, param0_name = common.sensitivity_parameters[i]
            param1, param1_name = common.sensitivity_parameters[j]
            for l in range(4):
                if l == 0:
                    m = (1, 1)
                    d = 0
                elif l == 1:
                    m = (-1, -1)
                    d = 0
                elif l == 2:
                    m = (1, -1)
                    d = 1
                elif l == 3:
                    m = (-1, 1)
                    d = 1
                for (k, n) in enumerate(parameters.parameter_sets.keys()):
                    r0[n][(param0, param1) + m] = r0_[k, ij, d, x0 : : m[0]]
            ij += 1
    return r0


def _get_sym(p, s):
    sign = '+' if s > 0 else '-'
    return '{}\\{}'.format(sign, p)


def _get_xlabel(params, signs):
    s = ','.join(itertools.starmap(_get_sym, zip(params, signs)))
    return '${}$'.format(s)


param_labels = {'rho': 'fecundity',
                'epsilon': 'encounters',
                'mu': 'mortality',
                'phi': 'feeding'}

columns_a = (('rho', 'epsilon'),
             ('mu', 'epsilon'),
             ('rho', 'phi'),
             ('mu', 'phi'))

rows_a = ((('Increased fitness\nIncreased movement',
            'e.g. mutualist'),
           ((+1, +1),
            (-1, +1),
            (+1, -1),
            (-1, -1))),
          (('Increased fitness\nDecreased movement',
            'e.g. mutualist'),
           ((+1, -1),
            (-1, -1),
            (+1, +1),
            (-1, +1))),
          (('Decreased fitness\nDecreased movement',
            'e.g. competitor, predator'),
           ((-1, -1),
            (+1, -1),
            (-1, +1),
            (+1, +1))),
          (('Decreased fitness\nIncreased movement',
            'e.g. competitor, predator'),
           ((-1, +1),
            (+1, +1),
            (-1, -1),
            (+1, -1))))

def plot_a(r0):
    arrow_pos = -0.12
    row_label_pos = -2.2
    figsize = (6, 5)
    fig, axes = pyplot.subplots(len(columns_a), len(rows_a),
                                sharey = True,
                                figsize = figsize,
                                squeeze = False)
    for (i, row) in enumerate(rows_a):
        row_labels, signs = row
        for (j, column) in enumerate(columns_a):
            params = column
            ax = axes[i, j]
            for (k, n) in enumerate(common.parameter_sets_ordered):
                ax.plot(r0[n][params + signs[j]], label = n,
                        alpha = common.alpha)
            xlabel = _get_xlabel(params, signs[j])
            ax.set_xlabel(xlabel,
                          horizontalalignment = 'right',
                          fontsize = 'x-small')
            ax.annotate('',
                        xy = (0.85, arrow_pos),
                        xytext = (0.52, arrow_pos),
                        xycoords = 'axes fraction',
                        annotation_clip = False,
                        arrowprops = dict(arrowstyle = '->',
                                          linewidth = 0.6))
            # ax.autoscale(tight = True)  # Bug!
            ax.set_yscale('log')
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
    for (ax, column) in zip(axes[-1, :], columns_a):
        params = column
        label = ' &\n'.join(param_labels[p].capitalize() for p in params)
        ax.annotate(label,
                    xy = (-0.05, -0.3),
                    xycoords = 'axes fraction',
                    annotation_clip = False,
                    horizontalalignment = 'left',
                    verticalalignment = 'top',
                    fontsize = 'small',
                    weight = 'bold')
    fig.text(0.68, 0.01, 'Pairings',
             horizontalalignment = 'center',
             verticalalignment = 'baseline',
             size = 'medium',
             weight = 'bold')
    for (ax, row) in zip(axes[:, 0], rows_a):
        row_labels, signs = row
        ax.annotate(row_labels[0],
                    xy = (row_label_pos, 0.55),
                    xycoords = 'axes fraction',
                    annotation_clip = False,
                    horizontalalignment = 'left',
                    verticalalignment = 'baseline',
                    fontsize = 'small',
                    weight = 'bold')
        ax.annotate(row_labels[1],
                    xy = (row_label_pos, 0.33),
                    xycoords = 'axes fraction',
                    annotation_clip = False,
                    horizontalalignment = 'left',
                    verticalalignment = 'baseline',
                    fontsize = 'small',
                    weight = 'normal')
    fig.text(0.05, 0.55,
             'Pathogen intrinsic growth rate',
             horizontalalignment = 'left',
             verticalalignment = 'center',
             rotation = 'vertical',
             size = 'medium',
             weight = 'bold')
    fig.tight_layout(rect = (0, 0, 1, 1))
    handles = [lines.Line2D([], [], color = c, alpha = common.alpha)
               for c in seaborn.color_palette()]
    labels = common.parameter_sets_ordered
    leg = fig.legend(handles, labels,
                     loc = 'lower left',
                     frameon = False,
                     numpoints = 1,
                     borderpad = 0,
                     borderaxespad = 0)
    if save:
        common.savefig(fig, append = '_a')
    return fig



def plot_bc(column_param, row_param, r0):
    signs = (+1, -1)
    arrow_pos = -0.075
    figsize = (4.5, 4)
    params = (column_param, row_param)
    fig, axes = pyplot.subplots(len(signs), len(signs),
                                sharey = True,
                                figsize = figsize,
                                squeeze = False)
    for (i, sign_row) in enumerate(signs):
        for (j, sign_col) in enumerate(signs):
            signs_ = (sign_row, sign_col)
            ax = axes[i, j]
            for (k, n) in enumerate(common.parameter_sets_ordered):
                ax.plot(r0[n][params + signs_], label = n,
                        alpha = common.alpha)
            xlabel = _get_xlabel(params, signs_)
            ax.set_xlabel(xlabel,
                          horizontalalignment = 'right',
                          fontsize = 'x-small')
            ax.annotate('',
                        xy = (0.85, arrow_pos),
                        xytext = (0.52, arrow_pos),
                        xycoords = 'axes fraction',
                        annotation_clip = False,
                        arrowprops = dict(arrowstyle = '->',
                                          linewidth = 0.6))
            # ax.autoscale(tight = True)  # Bug!
            ax.set_yscale('log')
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
    label_pre = {+1: 'Increased',
                 -1: 'Decreased'}
    for (ax, sign) in zip(axes[-1, :], signs):
        label = '{}\n{}'.format(label_pre[sign], param_labels[row_param])
        ax.annotate(label,
                    xy = (0.25, -0.25),
                    xycoords = 'axes fraction',
                    annotation_clip = False,
                    horizontalalignment = 'left',
                    verticalalignment = 'top',
                    weight = 'bold',
                    fontsize = 'small')
    for (ax, sign) in zip(axes[:, 0], signs):
        label = '{}\n{}'.format(label_pre[sign], param_labels[column_param])
        ax.annotate(label,
                    xy = (-0.55, 0.48),
                    xycoords = 'axes fraction',
                    annotation_clip = False,
                    horizontalalignment = 'left',
                    verticalalignment = 'center',
                    weight = 'bold',
                    fontsize = 'small')
    fig.text(0.02, 0.6,
             'Pathogen intrinsic growth rate',
             horizontalalignment = 'left',
             verticalalignment = 'center',
             rotation = 'vertical',
             size = 'medium',
             weight = 'bold')
    fig.tight_layout(rect = (0.05, 0.06, 1, 1))
    handles = [lines.Line2D([], [], color = c, alpha = common.alpha)
               for c in seaborn.color_palette()]
    labels = common.parameter_sets_ordered
    leg = fig.legend(handles, labels,
                     loc = (0, 0),
                     frameon = False,
                     numpoints = 1,
                     borderpad = 0,
                     borderaxespad = 0)
    return fig


def plot_b(r0):
    fig = plot_bc('rho', 'mu', r0)
    if save:
        common.savefig(fig, append = '_b')
    return fig


def plot_c(r0):
    fig = plot_bc('epsilon', 'phi', r0)
    if save:
        common.savefig(fig, append = '_c')
    return fig


if __name__ == '__main__':
    r0 = common.load_or_build_data(build)
    r0 = convert_to_dict(r0)
    plot_a(r0)
    plot_b(r0)
    plot_c(r0)
    pyplot.show()
