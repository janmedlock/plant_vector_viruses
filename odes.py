#!/usr/bin/python3

import collections

import ad
import numpy
import pandas
from scipy import integrate
from scipy import sparse

import seaborn_quiet as seaborn


def ODEs(Y, t, p):
    Vsm, Vim, Vsfs, Vifsp, Vifst, Vsfip, Vsfit, Vifi, Ps, Pi = Y
    Vm = Vsm + Vim
    Vsfi = Vsfip + Vsfit
    Vifs = Vifsp + Vifst
    Vsf = Vsfs + Vsfi
    Vif = Vifs + Vifi
    Vf = Vsf + Vif
    V = Vm + Vf
    P = Ps + Pi

    dVsm = (p.rho * Vf * (1 - Vf / p.kappa / P)
            - p.mu_m * Vsm
            - p.epsilon / (1 - p.phi) * Vsm
            + p.epsilon / p.phi * Vsf
            + p.gamma_m * Vim)

    dVim = (- p.mu_m * Vim
            - p.epsilon / (1 - p.phi) * Vim
            + p.epsilon / p.phi * Vif
            - p.gamma_m * Vim)

    dVsfs = (- p.mu_f * Vsfs
             + p.epsilon / (1 - p.phi) * Vsm * Ps / P
             - p.epsilon / p.phi * Vsfs
             + p.gamma_f * Vifs
             - p.betaP * Vifst / Ps * Vsfs)

    dVsfip = (- p.mu_f * Vsfip
              + p.epsilon / (1 - p.phi) * Vsm * Pi / P
              - p.epsilon / p.phi * Vsfip
              + p.gamma_f * Vifi
              + p.betaP * Vifst / Ps * Vsfs
              - p.alpha * Vsfip)

    dVsfit = (- p.mu_f * Vsfit
              - p.epsilon / p.phi * Vsfit
              + p.alpha * Vsfip
              - p.betaV * Vsfit)

    dVifsp = (- p.mu_f * Vifsp
              + p.epsilon / (1 - p.phi) * Vim * Ps / P
              - p.epsilon / p.phi * Vifsp
              - p.gamma_f * Vifsp
              - p.betaP * Vifst / Ps * Vifsp
              - p.alpha * Vifsp)

    dVifst = (- p.mu_f * Vifst
              - p.epsilon / p.phi * Vifst
              - p.gamma_f * Vifst
              - p.betaP * Vifst / Ps * Vifst
              + p.alpha * Vifsp)

    dVifi = (- p.mu_f * Vifi
             + p.epsilon / (1 - p.phi) * Vim * Pi / P
             - p.epsilon / p.phi * Vifi
             - p.gamma_f * Vifi
             + p.betaP * Vifst / Ps * Vifs
             + p.betaV * Vsfit)

    dPs = - p.betaP * Vifst

    dPi = p.betaP * Vifst

    return (dVsm, dVim, dVsfs, dVifsp, dVifst, dVsfip, dVsfit, dVifi, dPs, dPi)


def get_DFS0(p):
    Vsm0 = (1 - p.phi) * p.V0
    Vsfs0 = p.phi * p.V0
    Ps0 = p.P0
    Vim0 = 0
    Vifsp0 = 0
    Vifst0 = 0
    Vsfip0 = 0
    Vsfit0 = 0
    Vifi0 = 0
    Pi0 = 0
    DFS0 = (Vsm0, Vim0, Vsfs0, Vifsp0, Vifst0, Vsfip0, Vsfit0, Vifi0, Ps0, Pi0)
    return DFS0


def get_initial_conditions(p, vi0):
    Vsm0, _, Vsfs0, _, _, _, _, _, Ps0, _ = get_DFS0(p)
    Vim0 = vi0 * p.V0
    Vifsp0 = 0
    Vifst0 = 0
    Vsfip0 = 0
    Vsfit0 = 0
    Vifi0 = 0
    Pi0 = 0
    Vsm0 -= Vim0
    Vsfs0 -= Vifsp0 + Vifst0 + Vsfip0 + Vsfit0 + Vifi0
    Ps0 -= Pi0
    Y0 = (Vsm0, Vim0, Vsfs0, Vifsp0, Vifst0, Vsfip0, Vsfit0, Vifi0, Ps0, Pi0)
    return Y0


def Jacobian(Y, t, p):
    Y_ = ad.adnumber(numpy.asarray(Y))
    return numpy.asarray(ad.jacobian(ODEs(Y_, t, p), Y_))


def Jacobian_infected(Y, t, p):
    ix_i = [1, 3, 4, 5, 6, 7, 9]
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
    Y = integrate.odeint(ODEs, numpy.copy(Y0), t, args = (p, ))
    return pandas.DataFrame(Y,
                            columns = ('Vsm', 'Vim', 'Vsfs', 'Vifsp', 'Vifst',
                                       'Vsfip', 'Vsfit', 'Vifi', 'Ps', 'Pi'))


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
        ax.plot(t, Y[k] / N[k0], label = k)


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
