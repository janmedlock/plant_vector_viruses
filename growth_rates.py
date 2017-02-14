#!/usr/bin/python3

from matplotlib import pyplot
import numpy
import seaborn

import common
import odes
import parameters


save = True

figsize = (8.5, 5)


def get_growth_rate(p, t = common.tmax):
    DFS0 = odes.get_DFS0(p)
    if t > 0:
        T = numpy.linspace(0, t, 101)
        DFS = odes.solve(DFS0, T, p)
        DFS1 = DFS.iloc[-1]
    else:
        DFS1 = DFS0
    r, _ = odes.get_r_v_Jacobian(t, DFS1, p)
    return r


def get_pop_and_growth_rates(p, V0 = 0.1, t = 1000, Vmax = 1000):
    V0_old = p.V0
    p.V0 = V0
    DFS0 = odes.get_DFS0(p)
    p.V0 = V0_old
    T = numpy.linspace(0, t, 1001)
    Y = odes.solve(DFS0, T, p)
    pop = Y.iloc[:, : -2].sum(1)
    r, _ = zip(*(odes.get_r_v_Jacobian(t, row[1], p)
                 for (t, row) in zip(T, Y.iterrows())))
    ix = (pop <= Vmax)
    pop_ = numpy.hstack((0, pop[ix], Vmax))
    r_ = numpy.hstack((0, numpy.asarray(r)[ix], numpy.interp(Vmax, pop, r)))
    return (pop_, r_)


def main():
    fig, axes = pyplot.subplots(1, 1, figsize = figsize)
    for (n, p) in parameters.parameter_sets.items():
        pop, r = get_pop_and_growth_rates(p)
        axes.plot(pop, r, label = n, alpha = common.alpha)
    axes.set_xlabel('Initial vector population size')
    axes.set_ylabel('Pathogen intrinsic growth rate (d$^{-1}$)')
    axes.legend(loc = 'upper left')

    fig.tight_layout()

    if save:
        common.savefig(fig)

    return fig


if __name__ == '__main__':
    main()
    pyplot.show()
