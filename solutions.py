#!/usr/bin/python3

from matplotlib import pyplot
import numpy

import common
import odes
import parameters


def main():
    fig, axes = pyplot.subplots(2, 1,
                                sharex = True,
                                subplot_kw = dict(yscale = 'log'))
    for (ax, np) in zip(axes, parameters.parameter_sets.items()):
        n, p = np

        Y0 = odes.get_initial_conditions(p, common.vi0)
        Y = odes.solve(Y0, common.t, p)

        odes.plot_solution(ax, common.t, Y)

        ax.set_ylabel('Number')
        ax.set_xlim(common.t[0], common.t[-1])
        ax.set_title(n)
        if ax.is_last_row():
            ax.set_xlabel('Time (d)')
            ax.legend(loc = 'lower right', ncol = 2)

        common.style_axis(ax)

    common.savefig(fig)


if __name__ == '__main__':
    main()
    pyplot.show()
