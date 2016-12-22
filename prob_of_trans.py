#!/usr/bin/python3

import ad
import numpy
import pandas
from scipy import integrate
from scipy import sparse

import seaborn_quiet as seaborn


def ODEs(Y, t, p):
    Vm = Y[0]
    Vf = Y[1 : k + 1]
    Vt = Y[-2]
    T = Y[-1]

    dVm = (- p.sigmaV * Vm
           + p.tauV * numpy.sum(Vf)
           + p.tauV * Vt
           - p.gammaV * Vm
           - p.muV * (1 + p.deltaV) * Vm)

    dVf = (- p.tauV * Vf
           - p.gammaV * Vf
           - p.muV * Vf
           - psi * Vf)
    dVf[0] += p.sigmaV * Vm
    dVf[1 : ] += psi * Vf[ : -1]

    dVt = (psi * Vf[-1]
           - p.tauV * Vt
           - p.gammaV * Vt
           - p.muV * Vt
           - p.betaP * Vt)

    dT = p.betaP * Vt

    return numpy.hstack((dVm, dVf, dVt, dT))


def solve(t, p):
    Vm = 1
    Vf = [0 for _ in range(k)]
    Vt = 0
    T = 0
    Y0 = numpy.hstack((Vm, Vf, Vt, T))
    Y = integrate.odeint(ODEs, Y0, t, args = (p, ))
    return pandas.Series(Y[:, -1])


def plot_solution(ax, t, P, label):
    ax.plot(t, P, label = label)


if __name__ == '__main__':
    from matplotlib import pyplot

    import parameters

    t = numpy.linspace(0, 5, 101)
    k = 12
    psi = 50

    fig, ax = pyplot.subplots()

    seaborn.set_palette(seaborn.color_palette('Set1'))

    p_p = parameters.Persistent()
    P_p = solve(t, p_p)
    plot_solution(ax, t, P_p, 'Persistent')

    p_n = parameters.Nonpersistent()
    P_n = solve(t, p_n)
    plot_solution(ax, t, P_n, 'Non-persistent')

    ax.set_xlabel('Time (d)')
    ax.set_ylabel('Probability of having transmitted')
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc = 'lower right', ncol = 2)

    pyplot.show()
