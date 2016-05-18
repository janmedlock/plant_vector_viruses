#!/usr/bin/python3

from matplotlib import pyplot
import numpy
from scipy import optimize

import common
import odes
import parameters


def get_initial_conditions_round_1(p):
    '''
    Trying for approximate equilibrium, round 1.
    
    Find Pi0 that minimizes ||dY/dt||.
    '''
    # Guesses.
    Vms0 = p.QSSA.vm * (1 - common.vi0) * p.V0
    Vfs0 = p.QSSA.vf * (1 - common.vi0) * p.V0
    Vmi0 = p.QSSA.vm * common.vi0 * p.V0
    Vfi0 = p.QSSA.vf * common.vi0 * p.V0
    Pi0guess = 25 * common.vi0 * p.V0
    Ps0guess = p.P0 - Pi0guess

    def f(Pi0, p, Vms0, Vfs0, Vmi0, Vfi0):
        Ps0 = p.P0 - Pi0
        Y0 = (Vms0, Vfs0, Vmi0, Vfi0, Ps0, Pi0)
        dY0 = odes.ODEs(Y0, 0, p)
        return dY0[-1]

    res = optimize.root(f, Pi0guess, args = (p, Vms0, Vfs0, Vmi0, Vfi0))

    Pi0 = numpy.squeeze(res.x[0])
    Ps0 = p.P0 - Pi0

    Y0 = (Vms0, Vfs0, Vmi0, Vfi0, Ps0, Pi0)

    return Y0


def get_initial_conditions_round_2(p, Y0, t = 5):
    '''
    Trying for approximate equilibrium, round 2.

    Use eigenvector on the disease-free solution.
    '''

    V0 = numpy.sum(Y0[ : 4])
    Vi0 = numpy.sum(Y0[2 : 4])
    P0 = numpy.sum(Y0[4 : ])

    t_ = numpy.linspace(0, t, 101)
    DFS0 = odes.get_DFS0(p)
    DFS = odes.solve(DFS0, t_, p)
    r, v = odes.get_r_v_Jacobian(t_, DFS, p)
    i0 = v[-1, [2, 3, 5]]      # Infected compartments.
    vi0 = numpy.sum(i0[ : 2])  # Normalize so that Vmi0 + Vfi0 = 1.
    if numpy.abs(vi0) > 0:
        i0 /= vi0  # Normalize so that Vmi0 + Vfi0 = 1.
    else:
        assert numpy.allclose(i0, 0)

    # Use eigenvector for Vmi1, Vfi, Pi1.
    # Vmi1, Vfi1, Pi1 = i0 * Vi0
    # vm1 = Vmi1 / Vi0
    # Vms1 = vm1 * (V0 - Vi0)
    # Vfs1 = (1 - vm1) * (V0 - Vi0)

    # Use QSSA for Vmi1, Vfi1, and eigenvector for Pi1.
    Vmi1 = p.QSSA.vm * Vi0
    Vfi1 = p.QSSA.vf * Vi0
    Pi1 = i0[2] * Vi0
    Vms1 = p.QSSA.vm * (V0 - Vi0)
    Vfs1 = p.QSSA.vf * (V0 - Vi0)
    
    Ps1 = P0 - Pi1

    Y1 = (Vms1, Vfs1, Vmi1, Vfi1, Ps1, Pi1)
    return Y1


def get_initial_conditions_growth_rate(p):
    Y0 = get_initial_conditions_round_1(p)
    Y1 = get_initial_conditions_round_2(p, Y0)
    return Y1


def plot_growth_rates(ax, p):
    Y0 = get_initial_conditions_growth_rate(p)
    Y = odes.solve(Y0, common.t, p)

    r_QSSA = p.QSSA.r0(common.t)
    ax.plot(common.t, r_QSSA, label = 'QSSA', linestyle = 'dashed',
            zorder = 3)

    DFS0 = odes.get_DFS0(p)
    DFS = odes.solve(DFS0, common.t, p)
    r_J_DFS, _ = odes.get_r_v_Jacobian(common.t, DFS, p)
    ax.plot(common.t, r_J_DFS,
            label = ('Dominant eigenvalue of $\mathbf{J}$ along solution'
                     ' without pathogen'))

    r_J, _ = odes.get_r_v_Jacobian(common.t, Y, p)
    ax.plot(common.t, r_J,
            label = 'Dominant eigenvalue of $\mathbf{J}$ along solution')

    r_sol = odes.get_r_empirical(common.t, Y['Vmi'] + Y['Vfi'] + Y['Pi'])
    ax.plot(common.t, r_sol,
            label = 'From solution $I = V_{mi} + V_{fi} + P_i$')


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
