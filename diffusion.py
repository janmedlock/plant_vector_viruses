#!/usr/bin/python
#

import numpy
from scipy import integrate
from matplotlib import pyplot


def rhs(Y, t, Dxx):
    dY = Y * (1. - Y) + numpy.dot(Dxx, Y)
    return dY


def buildDxx(x):
    dx = x[1] - x[0]

    Dxx = numpy.zeros((len(x), ) * 2)

    Dxx += numpy.diag(- 2. / dx ** 2 * numpy.ones_like(x))
    Dxx += numpy.diag(  1. / dx ** 2 * numpy.ones_like(x[1 : ]),  1)
    Dxx += numpy.diag(  1. / dx ** 2 * numpy.ones_like(x[1 : ]), -1)

    # Dirichlet boundary conditions.
    # Dxx[0, -1] = 0.  # Left
    # Dxx[-1, 0] = 0.  # Right

    # Neumann boundary conditions.
    # Dxx[0, 1] = - Dxx[0, 0]      # Left
    # Dxx[-1, -2] = - Dxx[-1, -1]  # Right

    # Periodic boundary conditions.
    # Dxx[0, -1] = 1. / dx ** 2  # Left
    # Dxx[-1, 0] = 1. / dx ** 2  # Right

    # Right-moving wave.
    # Neumann boundary conditions on the left.
    Dxx[0, 1] = - Dxx[0, 0]
    # Dirichlet boundary condition on the right.
    Dxx[-1, 0] = 0.
    
    return Dxx


t = numpy.linspace(0, 10, 101)
x = numpy.linspace(-50, 50, 1001)

Y0 = numpy.where(x < 0., 1., 0.)

Y = integrate.odeint(rhs, Y0, t, args = (buildDxx(x), ))

(fig, ax) = pyplot.subplots(2, 1)
step = int(1 / (t[1] - t[0]))
ax[0].plot(x, Y[ : : step].T)
ax[0].set_ylabel('density')
ax[0].set_xlabel('distance')


def find_speed(t, x, Y, threshold = 0.01):
    locs = []
    for i in range(len(t)):
        # Y must be increasing, so use -Y.
        locs.append(numpy.interp(- threshold,
                                 - Y[i, len(x) // 2 : ],
                                 x[len(x) // 2 : ]))

    return numpy.diff(locs) / numpy.diff(t)


speeds = find_speed(t, x, Y)
cstar = 2

ax[1].plot(t[ : -1] + numpy.diff(t), speeds)
ax[1].axhline(cstar, color = 'black', linestyle = 'dotted')
ax[1].set_xlabel('time')
ax[1].set_ylabel('speed')

pyplot.show()
