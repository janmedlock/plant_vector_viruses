#!/usr/bin/python3

import copy
import re
import textwrap
import warnings

import ad
from matplotlib import pyplot
import numpy
from scipy import integrate
from scipy import optimize
from scipy import sparse

warnings.filterwarnings(
    'ignore',
    module = 'matplotlib',
    message = ('axes.color_cycle is deprecated '
               'and replaced with axes.prop_cycle; '
               'please use the latter.'))
import seaborn


class Parameters:
    sigmaV = 1 / 0.45 * 24
    tauV = 1 / 0.55 * 24
    gammaV = 1 / 4
    muVm = 0.02
    muVf = 0.01
    bVf = 0.08
    KV = 100
    betaV = 0.1 * 24
    betaP = 0.1 * 24
    gammaP = 0

    # Space not preceded by comma.
    _fill_regex = re.compile(r'[^,]( )')

    def __repr__(self):
        # Start with '<ClassName'
        s = '<{}'.format(self.__class__.__name__)
        params_strs = []
        for a in sorted(dir(self)):
            if not a.startswith('_'):
                v = getattr(self, a)
                if not callable(v):
                    params_strs.append('{} = {}'.format(a, v))
        if len(params_strs) > 0:
            s += ': '
            width = len(s)  # of '<ClassName: '
            s += ', '.join(params_strs)
        else:
            width = 0
        s += '>'
        # Wrap on commas and indent.
        wrapper = textwrap.TextWrapper(subsequent_indent = ' ' * width)
        nonspace = '*'  # A char that textwrap.TextWrapper won't linebreak on.
        # Replace spaces not preceded by commas with non-spaces.
        s_nonspaced = re.sub(r'(?<!,) ', nonspace, s)
        s_nonspaced_wrapped = wrapper.fill(s_nonspaced)
        # Replace non-spaces with spaces.
        s_wrapped = s_nonspaced_wrapped.replace(nonspace, ' ')
        return s_wrapped


p = Parameters()


def ODEs(Y, t, p):
    Vms, Vfs, Vmi, Vfi, Ps, Pi = Y
    V = Vms + Vfs + Vmi + Vfi
    P = Ps + Pi

    dVms = (- p.sigmaV * Vms + p.tauV * Vfs + p.gammaV * Vmi
            - p.muVm * Vms + p.bVf * (Vfs + Vfi) * (1 - V / p.KV / P))
    dVfs = (- p.betaV * Pi / P * Vfs + p.sigmaV * Vms - p.tauV * Vfs
            + p.gammaV * Vfi - p.muVf * Vfs)
    dVmi = (- p.sigmaV * Vmi + p.tauV * Vfi - p.gammaV * Vmi
            - p.muVm * Vmi)
    dVfi = (p.betaV * Pi / P * Vfs + p.sigmaV * Vmi - p.tauV * Vfi
            - p.gammaV * Vfi - p.muVf * Vfi)
    dPs = - p.betaP * Vfi * Ps / P + p.gammaP * Pi
    dPi = p.betaP * Vfi * Ps / P - p.gammaP * Pi

    return (dVms, dVfs, dVmi, dVfi, dPs, dPi)


def Jacobian(Y, t, p, func = ODEs):
    Y_ = ad.adnumber(Y)
    return ad.jacobian(func(Y_, t, p), Y_)


def get_dominant_eigenpair(A):
    # w, v = numpy.linalg.eig(A)
    # j = numpy.argmax(numpy.real(w))
    # return (w[j], v[:, j])
    w, v = sparse.linalg.eigs(numpy.asarray(A),
                              k = 1, which = 'LR',
                              maxiter = 10000)
    return map(numpy.squeeze, (w, v))


def get_r(Y, t, p):
    r = numpy.empty(len(t), dtype = numpy.complex_)
    for i in range(len(t)):
        w, _ = get_dominant_eigenpair(Jacobian(Y[i], t[i], p))
        r[i] = w
    return numpy.real_if_close(r)


def get_r_v(Y, t, p):
    r = numpy.empty(len(t), dtype = numpy.complex_)
    V = numpy.empty(numpy.shape(Y), dtype = numpy.complex_)
    for i in range(len(t)):
        w, v = get_dominant_eigenpair(Jacobian(Y[i], t[i], p))
        r[i] = w
        V[i] = v
    return map(numpy.real_if_close, (r, V))
    

def get_initial_conditions(p):
    '''
    Trying for approximate equilibrium.
    '''

    def f(Pi0, p, P0, Vms0, Vfs0, Vmi0, Vfi0):
        Ps0 = P0 - Pi0
        Y0 = (Vms0, Vfs0, Vmi0, Vfi0, Ps0, Pi0)
        dY0 = ODEs(Y0, 0, p)
        return dY0[-1]

    V0 = 100
    P0 = 10000
    vi0 = 0.01
    vs0 = 1 - vi0
    vm0 = p.tauV / (p.sigmaV + p.tauV)
    vf0 = p.sigmaV / (p.sigmaV + p.tauV)
    Vms0 = vm0 * vs0 * V0
    Vfs0 = vf0 * vs0 * V0
    Vmi0 = vm0 * vi0 * V0
    Vfi0 = vf0 * vi0 * V0
    Pi0 = 25 * vi0 * V0
    Ps0 = P0 - Pi0

    res = optimize.root(f, Pi0, args = (p, P0, Vms0, Vfs0, Vmi0, Vfi0))

    Y0 = (Vms0, Vfs0, Vmi0, Vfi0, Ps0, Pi0)

    return Y0


def plot_solution(ax, t, Y0, p):
    Y = integrate.odeint(ODEs, Y0, t, args = (p, ))

    Vms, Vfs, Vmi, Vfi, Ps, Pi = map(numpy.squeeze, numpy.hsplit(Y, 6))
    V = Vms + Vfs + Vmi + Vfi
    P = Ps + Pi
    Vi = Vmi + Vfi

    ax.semilogy(t, Ps, label = '$P_s$',
                    color = 'green', linestyle = 'solid')
    ax.semilogy(t, Vfs, label = '$V_{fs}$',
                    color = 'blue', linestyle = 'solid')
    ax.semilogy(t, Vms, label = '$V_{ms}$',
                    color = 'red', linestyle = 'solid')
    ax.semilogy(t, Pi, label = '$P_i$',
                    color = 'green', linestyle = 'dashed')
    ax.semilogy(t, Vfi, label = '$V_{fi}$',
                    color = 'red', linestyle = 'dashed')
    ax.semilogy(t, Vmi, label = '$V_{mi}$',
                    color = 'blue', linestyle = 'dashed')
    ax.set_ylabel('number infected')
    ax.legend(loc = 'lower right', ncol = 2)


def growth_rate_solution(t, Y0, p):
    Y = integrate.odeint(ODEs, Y0, t, args = (p, ))

    Vms, Vfs, Vmi, Vfi, Ps, Pi = map(numpy.squeeze, numpy.hsplit(Y, 6))
    V = Vms + Vfs + Vmi + Vfi
    P = Ps + Pi
    Vi = Vmi + Vfi

    r_P = numpy.diff(numpy.log(Pi)) / numpy.diff(t)
    r_V = numpy.diff(numpy.log(Vi)) / numpy.diff(t)

    return (r_P, r_V)


def growth_rate_Jacobian(t, Y0, p):
    DFE0 = numpy.array(Y0)
    # Make all uninfected.
    DFE0[0] += DFE0[2]
    DFE0[2] = 0
    DFE0[1] += DFE0[3]
    DFE0[3] = 0
    DFE0[4] += DFE0[5]
    DFE0[5] = 0

    DFE = integrate.odeint(ODEs, DFE0, t, args = (p, ))

    r = get_r(DFE, t, p)

    return r


def growth_rate_QSSA(t, Y0, p):
    vf = p.sigmaV / (p.sigmaV + p.tauV)

    muV = vf * p.muVf + (1 - vf) * p.muVm
    bV = vf * p.bVf + (1 - vf) * 0

    V0 = sum(Y0[ : 4])
    P = sum(Y0[4 : ])

    Vinf = (bV - muV) / bV * p.KV * P

    V = (V0 * numpy.exp((bV - muV) * t)
         / (1 + V0 / Vinf * (numpy.exp((bV - muV) * t) - 1)))

    # No logistic approximation
    # (i.e. K_V -> \infty => V_{\infty} \to \infty).
    # V = V0 * numpy.exp((p.bVf * vf - muV) * t)

    r = (- (muV + p.gammaV + p.gammaP)
         + numpy.sqrt((muV + p.gammaV + p.gammaP) ** 2
                      + 4 * (p.betaV * p.betaP * V / P * vf ** 2
                             - p.gammaP * (muV + p.gammaV)))) / 2

    return r


if __name__ == '__main__':
    Y0 = get_initial_conditions(p)

    tmax = 50
    t = numpy.linspace(0, tmax, 1001)

    # fig, ax = pyplot.subplots()
    # plot_solution(ax, t, Y0, p)


    fig, ax = pyplot.subplots()
    ax.set_xlabel('time (days)')
    ax.set_ylabel('$r$')

    # q = copy.copy(p)
    # q.KV = numpy.inf
    # r_QSSA_0 = growth_rate_QSSA(t, Y0, q)
    # ax.plot(t, r_QSSA_0, label = 'QSSA without logistic',
    #         linestyle = 'dashed', zorder = 10)

    r_QSSA = growth_rate_QSSA(t, Y0, p)
    ax.plot(t, r_QSSA, label = 'QSSA', linestyle = 'dashed', zorder = 2)

    # q = copy.copy(p)
    # q.KV = numpy.inf
    # r_J_0 = growth_rate_Jacobian(t, Y0, p)
    # ax.plot(t, r_J_0,
    #         label = 'Dominant eigenvalue of $J$ at DFE without logistic')

    r_J = growth_rate_Jacobian(t, Y0, p)
    ax.plot(t, r_J, label = 'Dominant eigenvalue of $\mathbf{J}$ at DFE')

    # q = copy.copy(p)
    # q.KV = numpy.inf
    # r_P_0, r_V_0 = growth_rate_solution(t, Y0, q)
    # ax.plot(t[1 : ], r_P_0,
    #         label = 'Solution to ODEs without logistic, $P_i$')
    # ax.plot(t[1 : ], r_V_0,
    #         label = 'Solution to ODEs without logistic, $V_i$')

    r_P, r_V = growth_rate_solution(t, Y0, p)
    ax.plot(t[1 : ], r_P, label = 'Solution to ODEs, $P_i$')
    ax.plot(t[1 : ], r_V, label = 'Solution to ODEs, $V_i$')

    ax.legend(loc = 'upper left')

    fig.savefig('Dropbox/growth_rates.pdf')

    pyplot.show()
