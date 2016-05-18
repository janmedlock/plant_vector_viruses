#!/usr/bin/python
#

import numpy
from scipy import integrate
from matplotlib import pyplot


class Params(object):
    def __repr__(self):
        components = ['{}:'.format(self.__class__.__name__)]
        for p in dir(self):
            if not p.startswith('_'):
                components.append('{} = {}'.format(p, getattr(self, p)))
        return '\n\t'.join(components)

params = Params()


def model_nondimensional(Y, t, p, Dxx):
    (V_S, V_I, P_I) = numpy.hsplit(Y, 3)

    V = V_S + V_I

    dV_S = (- P_I * V_S
            + p.gamma_V * V_I
            - p.mu_V * V_S
            + p.b_V * V * (1. - (1. - p.mu_V / p.b_V) * V)
            + numpy.dot(Dxx, V_S))

    dV_I = (+ P_I * V_S
            - p.gamma_V * V_I
            - p.mu_V * V_I
            + numpy.dot(Dxx, V_I))

    dP_I = p.beta_P * (1 - P_I) * V_I
    
    return numpy.hstack((dV_S, dV_I, dP_I))


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


# Pretty pictures.
# params.beta_V = 2.
# params.beta_P = 2.
# params.mu_V = 2.
# params.b_V = 2.
# params.K_V = numpy.inf
# params.D_V = 2. / 25. ** 2

# x = numpy.linspace(-0.2, 1., 101)
# t = numpy.linspace(0., 10., 11)

# BYDV Persistent
# beta_P = 0.5 / 2. * 24.  # 50% / 2 h converted to per day
beta_P = 0.10            # per day
beta_V = 0.5 / 3. * 24.  # 50% / 3 h converted to per day
gamma_V = 0.             # per day
mu_V = 0.03              # per day
b_V = 0.60               # per day
K_V = 1000.              # per meter
# Einstein's formula: D = 2 * dx ** 2 / dt
# dx = 6 cm in dt = 1 h
D_V = 2. * (6. / 100.) ** 2 / (1. / 24.)  # meter^2 per day
P = 1                    # per meter


# Non-dimensional parameters
chi_V = K_V * (1. - mu_V / b_V)
chi_P = P
chi_t = 1. / beta_V
chi_x = numpy.sqrt(1. / chi_t / D_V)

params.beta_P = beta_P * chi_V / chi_P * chi_t

params.beta_V = beta_V * chi_t
params.gamma_V = gamma_V * chi_t
params.b_V = b_V * chi_t
params.mu_V = mu_V * chi_t

# Wave speed
q1 = (params.gamma_V + params.mu_V) ** 2 + 3 * params.beta_P
q2 = ((params.gamma_V + params.mu_V)
      * (2 * (params.gamma_V + params.mu_V) ** 2 + 9 * params.beta_P))
q3 = ((params.gamma_V + params.mu_V) ** 2 + 4 * params.beta_P)
cstar = numpy.sqrt((2 * q1 ** (3. / 2) - q2) / q3)

t_dim = numpy.linspace(0, 10, 101)        # days
x_dim = numpy.linspace(-1000, 1000, 1001) # meters

t = t_dim / chi_t
x = x_dim / chi_x

# One infected vector in center.
# V_I0 = numpy.zeros_like(x)
# V_I0[len(x) // 2] = 1.
# P_I0 = numpy.zeros_like(x)

# Right-moving wave.
V_I0 = numpy.where(x < 0., 1 / (1 + params.gamma_V + params.mu_V), 0.)
P_I0 = numpy.where(x < 0., 1., 0.)

V_S0 = 1. - V_I0

Y0 = numpy.hstack((V_S0, V_I0, P_I0))
Y = integrate.odeint(model_nondimensional, Y0, t,
                     args = (params, buildDxx(x)))
(V_S, V_I, P_I) = numpy.hsplit(Y, 3)


V_S_dim = V_S * chi_V
V_I_dim = V_I * chi_V
P_I_dim = P_I * chi_P

V_dim = V_S_dim + V_I_dim


(fig, ax) = pyplot.subplots(3, 1, sharex = True)
step = int(1 / (t_dim[1] - t_dim[0]))
ax[0].plot(x_dim, V_S_dim[ : : step].T)
ax[1].plot(x_dim, V_I_dim[ : : step].T)
ax[2].plot(x_dim, P_I_dim[ : : step].T)
ax[0].set_ylabel('density of susceptible vectors')
ax[1].set_ylabel('density of infected vectors')
ax[2].set_ylabel('density of infected plants')
ax[2].set_xlabel('distance (m)')
ax[0].set_ylim(ymin = 0)
ax[1].set_ylim(ymin = 0)
ax[2].set_ylim(ymin = 0)


def find_speed(t, x, Y, threshold = 0.01):
    locs = []
    for i in range(len(t)):
        # Y must be increasing, so use -Y.
        locs.append(numpy.interp(- threshold,
                                 - Y[i, len(x) // 2 : ],
                                 x[len(x) // 2 : ]))

    return numpy.diff(locs) / numpy.diff(t)


speeds_dim = find_speed(t_dim, x_dim, P_I_dim)
cstar_dim = cstar * chi_x / chi_t

(fig, ax) = pyplot.subplots(1, 1, sharex = True)
ax.plot(t_dim[ : -1] + numpy.diff(t_dim), speeds_dim)
ax.axhline(cstar_dim, color = 'black', linestyle = 'dotted')
ax.set_xlabel('time (days)')
ax.set_ylabel('speed of invasion of virus into plants (meters per day)')

pyplot.show()
