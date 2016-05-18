#!/usr/bin/python3

import numpy
from scipy import integrate
from matplotlib import pyplot


def RHS(V, t, tau, sigma):
    V_F, V_M, V_G = V
    dV_F = - (1 + tau) * V_F
    dV_M = tau * V_F - (1 + sigma) * V_M
    dV_G = sigma * V_M - V_G

    return (dV_F, dV_M, dV_G)


tau = sigma = 10

t = numpy.linspace(0, 1, 101)

V0 = (1, 0, 0)

V = integrate.odeint(RHS, V0, t, (tau, sigma))

V_F, V_M, V_G = map(numpy.squeeze, numpy.hsplit(V, 3))

pyplot.plot(t, sigma * V_M)

pyplot.show()

