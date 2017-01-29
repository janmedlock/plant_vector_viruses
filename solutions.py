#!/usr/bin/python3

from matplotlib import pyplot
import numpy
import pandas

import common
import odes
import parameters
import seaborn_quiet as seaborn


def main():
    fig, axes = pyplot.subplots(2, 2, sharex = 'col', sharey = 'row')
    for (ax, np) in zip(axes.T, parameters.parameter_sets.items()):
        n, p = np

        t0 = common.t
        DFS0 = odes.get_DFS0(p)
        DFS = odes.solve(DFS0, t0, p)

        t1 = t0[-1] + common.t
        Y0 = DFS.iloc[-1].copy()
        Y0['Vmi'] = 1
        Y0['Vms'] -= Y0['Vmi']
        Y = odes.solve(Y0, t1, p)

        t = numpy.hstack((t0, t1))
        Z = pandas.concat([DFS, Y])

        style = dict(alpha = 0.7)
        style_s = dict(linestyle = 'solid')
        style_s.update(style)
        style_i = dict(linestyle = 'dashed')
        style_i.update(style)

        colors = seaborn.color_palette('Set1', 3)

        ax[0].plot(t, Z['Vms'], label = '$V_{ms}$', color = colors[0],
                   **style_s)
        ax[0].plot(t, Z['Vfs'], label = '$V_{fs}$', color = colors[1],
                   **style_s)
        ax[0].plot(t, Z['Vmi'], label = '$V_{mi}$', color = colors[0],
                   **style_i)
        ax[0].plot(t, Z['Vfi'], label = '$V_{fi}$', color = colors[1],
                   **style_i)
        ax[1].plot(t, Z['Ps'], label = '$P_s$', color = colors[2],
                   **style_s)
        ax[1].plot(t, Z['Pi'], label = '$P_i$', color = colors[2],
                   **style_i)

        ax[0].set_title(n)
        ax[1].set_xlabel('Time (d)')
        if ax[0].is_first_col():
            ax[0].set_ylabel('Vectors')
            ax[1].set_ylabel('Plants')
            ax[0].legend(loc = 'upper left', ncol = 2)
            ax[1].legend(loc = 'upper left', ncol = 2)

        common.style_axis(ax[0])
        common.style_axis(ax[1])

    common.savefig(fig)


if __name__ == '__main__':
    main()
    pyplot.show()
