#!/usr/bin/python3

from matplotlib import pyplot
import numpy
import pandas

import common
import odes
import parameters
import seaborn_quiet as seaborn


save = True
alpha = 0.7
colors = seaborn.color_palette('Set1', 5)
colors = [colors[ix] for ix in (0, 1, 3, 4, 2)]



def main():
    fig, axes = pyplot.subplots(2, 2, sharex = 'col', sharey = 'row')
    for (ax, np) in zip(axes.T, parameters.parameter_sets.items()):
        n, p = np
        t = common.t
        Y0 = odes.get_initial_conditions(p, common.vi0)
        Y = odes.solve(Y0, t, p)

        style = dict(alpha = alpha)
        style_s = dict(linestyle = 'solid')
        style_s.update(style)
        style_i = dict(linestyle = 'dashed')
        style_i.update(style)

        ax[0].plot(t, Y['Vsm'], label = '$V_{sm}$',
                   color = colors[0], **style_s)
        ax[0].plot(t, Y['Vsfs'], label = '$V_{sfs}$',
                   color = colors[1], **style_s)
        ax[0].plot(t, Y['Vsfip'], label = '$V_{sfip}$',
                   color = colors[2], **style_s)
        ax[0].plot(t, Y['Vsfit'], label = '$V_{sfit}$',
                   color = colors[3], **style_s)
        ax[0].plot(t, Y['Vim'], label = '$V_{im}$',
                   color = colors[0], **style_i)
        ax[0].plot(t, Y['Vifi'], label = '$V_{ifi}$',
                   color = colors[1], **style_i)
        ax[0].plot(t, Y['Vifsp'], label = '$V_{ifsp}$',
                   color = colors[2], **style_i)
        ax[0].plot(t, Y['Vifst'], label = '$V_{ifst}$',
                   color = colors[3], **style_i)
        ax[1].plot(t, Y['Ps'], label = '$P_s$',
                   color = colors[4], **style_s)
        ax[1].plot(t, Y['Pi'], label = '$P_i$',
                   color = colors[4], **style_i)

        ax[0].set_title(n)
        ax[1].set_xlabel('Time (d)')
        if ax[0].is_first_col():
            ax[0].set_ylabel('Vectors')
            ax[1].set_ylabel('Plants')
            ax[0].legend(loc = 'upper left', ncol = 2)
            ax[1].legend(loc = 'upper left', ncol = 1)

        common.style_axis(ax[0])
        common.style_axis(ax[1])

    fig.tight_layout()

    if save:
        common.savefig(fig)

    return fig


if __name__ == '__main__':
    main()
    pyplot.show()
