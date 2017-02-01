#!/usr/bin/python3

import collections

import joblib
from matplotlib import pyplot
import numpy
from scipy import optimize

import common
import odes
import parameters

import seaborn_quiet as seaborn


save = True

figsize = (8.5, 5)
seaborn.set_palette('Dark2')
alpha = 0.7


def get_pop_and_growth_rates(p):
    DFS0 = odes.get_DFS0(p)
    Y = odes.solve(DFS0, common.t, p)
    pop = Y.iloc[:, : -2].sum(1)
    r, _ = zip(*(odes.get_r_v_Jacobian(t, row[1], p)
                 for (t, row) in zip(common.t, Y.iterrows())))
    return (pop, numpy.asarray(r))


def main():
    with joblib.parallel.Parallel(n_jobs = -1) as parallel:
        pop_r = parallel(joblib.delayed(get_pop_and_growth_rates)(p)
                         for p in parameters.parameter_sets.values())
    pop, r = zip(*pop_r)

    fig = pyplot.figure(figsize = figsize)
    axes_popvstime = fig.add_subplot(2, 2, 1)
    for (n, pop_) in zip(parameters.parameter_sets.keys(), pop):
        # All parameter sets have the same vector population size.
        axes_popvstime.plot(common.t, pop_,
                            color = 'k', alpha = alpha)
        break
    axes_popvstime.set_ylabel('Vector population size')
    common.style_axis(axes_popvstime)
    for label in axes_popvstime.get_xticklabels():
        label.set_visible(False)
    axes_popvstime.xaxis.offsetText.set_visible(False)

    axes_growthvstime = fig.add_subplot(2, 2, 3)
    for (n, r_) in zip(parameters.parameter_sets.keys(), r):
        axes_growthvstime.plot(common.t, r_,
                               label = n, alpha = alpha)
    axes_growthvstime.set_xlabel('Time (d)')
    axes_growthvstime.set_ylabel('Infection growth rate (d$^{-1}$)')
    common.style_axis(axes_growthvstime)

    axes_growthvspop = fig.add_subplot(1, 2, 2)
    for (n, r_, pop_) in zip(parameters.parameter_sets.keys(), r, pop):
        axes_growthvspop.plot(pop_, r_,
                              label = n, alpha = alpha)
    axes_growthvspop.set_xlabel('Vector population size')
    axes_growthvspop.set_ylabel('Infection growth rate (d$^{-1}$)')
    axes_growthvspop.legend(loc = 'upper left')

    fig.tight_layout()

    if save:
        common.savefig(fig)

    return fig


if __name__ == '__main__':
    main()
    pyplot.show()
