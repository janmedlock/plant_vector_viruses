#!/usr/bin/python3

import collections

import ad
import numpy
import pandas
from scipy import integrate
from scipy import sparse

import seaborn_quiet as seaborn


def ODEs(Y, t, p):
    Vsm, Vim, Vsfs, Vifs, Vsfi, Vifi, Ps, Pi = Y
    Vm = Vsm + Vim
    Vsf = Vsfs + Vsfi
    Vif = Vifs + Vifi
    Vf = Vsf + Vif
    V = Vm + Vf
    P = Ps + Pi

    dVsm = (p.bV * Vf * (1 - V / p.KV / P)
            - p.muV * (1 + p.deltaV) * Vsm
            - p.fV / (1 - p.phiV) * Vsm
            + p.fV / p.phiV * Vsf
            + p.gammaV * Vim)

    dVim = (- p.muV * (1 + p.deltaV) * Vim
            - p.fV / (1 - p.phiV) * Vim
            + p.fV / p.phiV * Vif
            - p.gammaV * Vim)

    dVsfs = (- p.muV * Vsfs
             + p.fV / (1 - p.phiV) * Vsm * Ps / P
             - p.fV / p.phiV * Vsfs
             + p.gammaV * Vifs
             + p.gammaP * Vsfi
             - p.betaP * Vifs / Ps * Vsfs)

    dVsfi = (- p.muV * Vsfi
             + p.fV / (1 - p.phiV) * Vsm * Pi / P
             - p.fV / p.phiV * Vsfi
             + p.gammaV * Vifi
             - p.gammaP * Vsfi
             + p.betaP * Vifs / Ps * Vsfs
             - p.betaV * Vsfi)

    dVifs = (- p.muV * Vifs
             + p.fV / (1 - p.phiV) * Vim * Ps / P
             - p.fV / p.phiV * Vifs
             - p.gammaV * Vifs
             + p.gammaP * Vifi
             - p.betaP * Vifs / Ps * Vifs)

    dVifi = (- p.muV * Vifi
             + p.fV / (1 - p.phiV) * Vim * Pi / P
             - p.fV / p.phiV * Vifi
             - p.gammaV * Vifi
             - p.gammaP * Vifi
             + p.betaP * Vifs / Ps * Vifs
             + p.betaV * Vsfi)

    dPs = - p.betaP * Vifs + p.gammaP * Pi

    dPi = p.betaP * Vifs - p.gammaP * Pi

    return (dVsm, dVim, dVsfs, dVifs, dVsfi, dVifi, dPs, dPi)


def get_DFS0(p):
    Vsm0 = (1 - p.phiV) * p.V0
    Vim0 = 0
    Vsfs0 = p.phiV * p.V0
    Vifs0 = 0
    Vsfi0 = 0
    Vifi0 = 0
    Ps0 = p.P0
    Pi0 = 0
    DFS0 = (Vsm0, Vim0, Vsfs0, Vifs0, Vsfi0, Vifi0, Ps0, Pi0)
    return DFS0


def get_initial_conditions(p, vi0):
    Vsm0, _, Vsfs0, _, _, _, Ps0, _ = get_DFS0(p)
    Vim0 = vi0 * p.V0
    Vifs0 = 0
    Vsfi0 = 0
    Vifi0 = 0
    Pi0 = 0
    Vsm0 -= Vim0
    Vsfs0 -= Vifs0
    Ps0 -= Pi0
    Y0 = (Vsm0, Vim0, Vsfs0, Vifs0, Vsfi0, Vifi0, Ps0, Pi0)
    return Y0


def Jacobian(Y, t, p):
    Y_ = ad.adnumber(numpy.asarray(Y))
    return numpy.asarray(ad.jacobian(ODEs(Y_, t, p), Y_))


def Jacobian_infected(Y, t, p):
    ix_i = [1, 3, 4, 5, 7]
    Y_ = ad.adnumber(numpy.asarray(Y))
    J = ad.jacobian(ODEs(Y_, t, p), Y_)
    J_i = [[J[r][c] for c in ix_i] for r in ix_i]
    return numpy.asarray(J_i)


def get_r_v_Jacobian(t, Y, p, use_sparse_solver = False):
    J = Jacobian_infected(Y, t, p)
    # Get the dominant eigenpair.
    if use_sparse_solver:
        W, V = sparse.linalg.eigs(numpy.asarray(J),
                                  k = 1, which = 'LR',
                                  maxiter = 10000)
        j = 0
    else:
        W, V = numpy.linalg.eig(J)
        j = numpy.argmax(numpy.real(W))
    r = W[j]
    v = V[:, j]
    # Normalize
    for i in range(len(v)):
        if not numpy.isclose(v[i], 0):
            v /= v[i]
            break
    return map(numpy.real_if_close, (r, v))
    

def get_r_empirical(t, x, n = 1):
    if n == 0:
        n = len(t)

    logx = numpy.log(x.values)

    dlogx = numpy.empty(numpy.shape(x))
    dt = numpy.empty(numpy.shape(t))

    # (n+1)-point difference: dy[i] = y[i] - y[i - n]
    dlogx[n : ] = logx[n : ] - logx[ : - n]
    dt[n : ] = t[n : ] - t[ : - n]

    # (i+1)-point difference for i < n: dy[i] = y[i] - y[0]
    dlogx[ : n] = logx[ : n] - logx[0]
    dt[ : n] = t[ : n] - t[0]

    # Expand dt to handle multi-dimensional x.
    for _ in range(numpy.ndim(x) - 1):
        dt = dt[..., numpy.newaxis]

    # r = dlogx / dt
    r = numpy.ma.divide(dlogx, dt)
    return r


def solve(Y0, t, p):
    Y = integrate.odeint(ODEs, Y0, t, args = (p, ))
    return pandas.DataFrame(Y,
                            columns = ('Vsm', 'Vim', 'Vsfs', 'Vifs',
                                       'Vsfi', 'Vifi', 'Ps', 'Pi'))


def plot_solution(ax, t, Y):
    # Population sizes.
    N = collections.Counter()
    for (k, v) in Y.items():
        k0 = k[0]
        N[k0] += v

    # Reorder to get legend right.
    cols = list(Y.columns)
    cols = cols[0 : : 2] + cols[1 : : 2]
    for k in cols:
        k0 = k[0]
        ax.plot(t, Y[k] / N[k0], **common.style[k])


if __name__ == '__main__':
    from matplotlib import pyplot

    import common
    import parameters

    p = parameters.Persistent()

    Y0 = get_initial_conditions(p, common.vi0)
    Y = solve(Y0, common.t, p)

    fig, ax = pyplot.subplots()

    plot_solution(ax, common.t, Y)

    ax.set_xlabel('Time (d)')
    ax.set_ylabel('Proportion')
    ax.set_xlim(common.t[0], common.t[-1])
    ax.set_ylim(0, 1)
    ax.legend(loc = 'upper right', ncol = 2)

    pyplot.show()
