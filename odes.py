#!/usr/bin/python3

import ad
import numpy
import pandas
from scipy import integrate
from scipy import sparse

import seaborn_quiet as seaborn


def ODEs(Y, t, p):
    Vms, Vfs, Vmi, Vfi, Ps, Pi = Y
    V = Vms + Vfs + Vmi + Vfi
    P = Ps + Pi

    dVms = (- p.sigmaV * Vms + p.tauV * Vfs + p.gammaV * Vmi
            - p.muV * (1 + p.deltaV) * Vms
            + p.bV * (Vfs + Vfi) * (1 - V / p.KV / P))
    dVfs = (- p.betaV * Pi / P * Vfs + p.sigmaV * Vms - p.tauV * Vfs
            + p.gammaV * Vfi - p.muV * Vfs)
    dVmi = (- p.sigmaV * Vmi + p.tauV * Vfi - p.gammaV * Vmi
            - p.muV * (1 + p.deltaV) * Vmi)
    dVfi = (p.betaV * Pi / P * Vfs + p.sigmaV * Vmi - p.tauV * Vfi
            - p.gammaV * Vfi - p.muV * Vfi)
    dPs = - p.betaP * Vfi * Ps / P + p.gammaP * Pi
    dPi = p.betaP * Vfi * Ps / P - p.gammaP * Pi

    return (dVms, dVfs, dVmi, dVfi, dPs, dPi)


def get_initial_conditions(p, vi0):
    Vmi0 = vi0 * p.V0
    Vfi0 = 0
    Vms0 = p.QSSA.vm * p.V0 - Vmi0
    Vfs0 = p.QSSA.vf * p.V0 - Vfi0
    Pi0 = 0
    Ps0 = p.P0 - Pi0
    Y0 = (Vms0, Vfs0, Vmi0, Vfi0, Ps0, Pi0)
    return Y0


def get_DFS0(p):
    Vmi0 = 0
    Vfi0 = 0
    Vms0 = p.QSSA.vm * p.V0
    Vfs0 = p.QSSA.vf * p.V0
    Pi0 = 0
    Ps0 = p.P0
    DFS0 = (Vms0, Vfs0, Vmi0, Vfi0, Ps0, Pi0)
    return DFS0


def Jacobian(Y, t, p):
    Y_ = ad.adnumber(numpy.asarray(Y))
    return numpy.asarray(ad.jacobian(ODEs(Y_, t, p), Y_))


def Jacobian_infected(Y, t, p):
    ix_i = [2, 3, 5]
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
                            columns = ('Vms', 'Vfs', 'Vmi', 'Vfi', 'Ps', 'Pi'))


def plot_solution(ax, t, Y):
    style = dict(alpha = 0.7)
    style_s = dict(linestyle = 'solid')
    style_s.update(style)
    style_i = dict(linestyle = 'dashed')
    style_i.update(style)

    colors = seaborn.color_palette('Set1', 3)

    V = Y['Vms'] + Y['Vfs'] + Y['Vmi'] + Y['Vfi']
    P = Y['Ps'] + Y['Pi']
    ax.plot(t, Y['Vms'] / V, label = '$V_{ms}$', color = colors[0],
            **style_s)
    ax.plot(t, Y['Vfs'] / V, label = '$V_{fs}$', color = colors[1],
            **style_s)
    ax.plot(t, Y['Ps'] / P, label = '$P_s$', color = colors[2],
            **style_s)
    ax.plot(t, Y['Vmi'] / V, label = '$V_{mi}$', color = colors[0],
            **style_i)
    ax.plot(t, Y['Vfi'] / V, label = '$V_{fi}$', color = colors[1],
            **style_i)
    ax.plot(t, Y['Pi'] / P, label = '$P_i$', color = colors[2],
            **style_i)


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
    ax.legend(loc = 'lower right', ncol = 2)

    pyplot.show()
