#!/usr/bin/python3

from matplotlib import pyplot
import numpy
from scipy import optimize

import common
import odes
import parameters


def get_growth_rate(p, t = 150):
    T = numpy.linspace(0, t, 101)
    DFS0 = odes.get_DFS0(p)
    DFS = odes.solve(DFS0, T, p)
    DFS1 = DFS.values[-1]
    r, _ = odes.get_r_v_Jacobian(t, DFS1, p)

    # r_QSSA = p.QSSA.r0(t)

    return r


def plot_growth_rates(ax, p):
    DFS0 = odes.get_DFS0(p)
    DFS = odes.solve(DFS0, common.t, p)
    r, _ = zip(*(odes.get_r_v_Jacobian(t, row[1].values, p)
                 for (t, row) in zip(common.t, DFS.iterrows())))
    ax.plot(common.t, r,
            label = ('Numerical solution of ODEs'))

    r_QSSA = p.QSSA.r0(common.t)
    ax.plot(common.t, r_QSSA, label = 'QSSA', linestyle = 'dashed')


def main():
    fig, axes = pyplot.subplots(2, 1,
                                sharex = True)

    for (ax, np) in zip(axes, parameters.parameter_sets.items()):
        n, p = np

        plot_growth_rates(ax, p)

        ax.set_ylabel('Infection growth rate (d$^{-1}$)')
        ax.set_xlim(common.t[0], common.t[-1])
        ax.set_title(n)
        if ax.is_first_row():
            ax.legend(loc = 'upper left')
        if ax.is_last_row():
            ax.set_xlabel('Time (d)')

        common.style_axis(ax)

    common.savefig(fig)


if __name__ == '__main__':
    main()
    pyplot.show()
