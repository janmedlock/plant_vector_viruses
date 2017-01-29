#!/usr/bin/python3

from matplotlib import pyplot
import numpy
from scipy import optimize

import common
import odes
import parameters

import seaborn_quiet as seaborn


save = True

figsize = (8.5, 3)
seaborn.set_palette('Dark2')
alpha = 0.7


def get_growth_rate(p, t = 150):
    DFS0 = odes.get_DFS0(p)
    if t > 0:
        T = numpy.linspace(0, t, 101)
        DFS = odes.solve(DFS0, T, p)
        DFS1 = DFS.iloc[-1]
    else:
        DFS1 = DFS0
    r, _ = odes.get_r_v_Jacobian(t, DFS1, p)

    # r_QSSA = p.QSSA.r0(t)

    return r


def plot_growth_rates(ax, p, n):
    DFS0 = odes.get_DFS0(p)
    DFS = odes.solve(DFS0, common.t, p)
    r, _ = zip(*(odes.get_r_v_Jacobian(t, row[1], p)
                 for (t, row) in zip(common.t, DFS.iterrows())))
    ax.plot(common.t, r, label = n, alpha = alpha)


def main():
    fig, axes = pyplot.subplots(1, 1,
                                sharex = True,
                                figsize = figsize)

    for np in parameters.parameter_sets.items():
        n, p = np
        plot_growth_rates(axes, p, n)

    axes.set_xlabel('Time (d)')
    axes.set_ylabel('Infection growth rate (d$^{-1}$)')
    axes.set_xlim(common.t[0], common.t[-1])
    axes.legend(loc = 'upper left')

    common.style_axis(axes)

    fig.tight_layout()

    if save:
        common.savefig(fig)

    return fig


if __name__ == '__main__':
    main()
    pyplot.show()
