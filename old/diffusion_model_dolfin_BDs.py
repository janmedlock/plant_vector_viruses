#!/usr/bin/python
#

import dolfin
import numpy
from matplotlib import pyplot


# BYDV Persistent
# beta_P = 0.5 / 2. * 24.  # 50% / 2 h converted to per day
beta_P = 0.10            # per day
beta_V = 0.5 / 3. * 24.  # 50% / 3 h converted to per day
mu_V = 0.03              # per day
b_V = 0.60               # per day
K_V = 1000.              # per meter
# Einstein's formula: D = 2 * dx ** 2 / dt
# dx = 6 cm in dt = 1 h
D_V = 2. * (6. / 100.) ** 2 / (1. / 24.)  # meter^2 per day
P = 1                    # per meter


x_max = 100.
nx = 1000
t_max = 10.
nt = 101

t = numpy.linspace(0., 10., nt)
dt = t[1] - t[0]

mesh = dolfin.IntervalMesh(nx, - x_max, x_max)
x = mesh.coordinates()[:, 0]


S = dolfin.FunctionSpace(mesh, 'CG', 1)


dolfin.set_log_level(dolfin.ERROR)


variables = ['V_S', 'V_I', 'P_I']


# Boundary conditions
def boundary_right(X, on_boundary):
    tol = 1e-14
    return on_boundary and abs(X[0] - x[-1]) < tol

# Neumann on the left: no explicit BC there.
bc_right = dolfin.DirichletBC(S, dolfin.Constant(0), boundary_right)
# Same BCs for all components.
bcs = {k: bc_right for k in variables}


# Initial condition
# Right-moving wave.
Vstar = K_V * (1. - mu_V / b_V)
V_Istar = Vstar * beta_V / (mu_V + beta_V)

IC = {'V_S': dolfin.Expression('x[0] < 0. ? {} : {}'.format(Vstar - V_Istar,
                                                           Vstar)),
      'V_I': dolfin.Expression('x[0] < 0. ? {} : {}'.format(V_Istar, 0.)),
      'P_I': dolfin.Expression('x[0] < 0. ? {} : {}'.format(P, 0.))}


Y0 = {k: dolfin.Function(S) for k in variables}
Y1 = {k: dolfin.Function(S) for k in variables}
Phi = {k: dolfin.TestFunction(S) for k in variables}


def reaction(Y):
    V_S = Y['V_S']
    V_I = Y['V_I']
    P_I = Y['P_I']

    return {
        'V_S': (b_V * (V_S + V_I) * (1 - (V_S + V_I) / K_V)  # birth
                - mu_V * V_S                                 # death
                - beta_V * P_I / P * V_S),                   # infection
        'V_I': (- mu_V * V_I                                 # death
                + beta_V * P_I / P * V_S),                   # infection
        'P_I': beta_P * V_I * (1. - P_I / P)                 # infection
    }


# Diffusion coefficients.
D = {'V_S': D_V,
     'V_I': D_V,
     'P_I': 0}

def diffusion(Y):
    return {k: - D[k] * dolfin.inner(dolfin.nabla_grad(Y[k]),
                                     dolfin.nabla_grad(Phi[k]))
            for k in variables}


def RHS(Y):
    reaction_ = reaction(Y)
    diffusion_ = diffusion(Y)
    return {k: reaction_[k] * Phi[k] + diffusion_[k]
            for k in variables}


# Backward difference.
RHS1 = RHS(Y1)
F = {k: ((Y1[k] - Y0[k]) / dt * Phi[k] - RHS1[k]) * dolfin.dx
     for k in variables}


# Set initial condition.
for k in variables:
    Y0[k].interpolate(IC[k])

sp = {'newton_solver': {'maximum_iterations': 100}}
sol= {k: numpy.zeros((len(t), mesh.num_cells() + 1))
      for k in variables}
for k in variables:
    sol[k][0] = dolfin.project(Y0[k]).vector().array()
for i in range(1, len(t)):
    # Find Y1.
    for k in variables:
        dolfin.solve(F[k] == 0, Y1[k], bcs[k], solver_parameters = sp)
        # It seems more stable when multiplied by dt.
        # dolfin.solve(dt * F[k] == 0, Y1[k], bcs[k], solver_parameters = sp)

    # Set Y0 = Y1 and store.
    for k in variables:
        Y0[k].assign(Y1[k])
        sol[k][i] = dolfin.project(Y0[k]).vector().array()


threshold = 0.01
position = {k: numpy.zeros(len(t)) for k in variables}
for i in range(len(t)):
    for k in variables:
        position[k][i] = numpy.interp(- threshold, - sol[k][i], x)
speed = {k: numpy.diff(position[k]) / numpy.diff(t)
         for k in variables}

q0 = beta_P * beta_V * Vstar / P
q1 = mu_V ** 2 + 3 * q0
q2 = 2 * mu_V ** 2 + 9 * q0
q3 = mu_V ** 2 + 4 * q0
cstar = numpy.sqrt(D_V * (2 * q1 ** (3. / 2) - mu_V * q2) / q3)

(fig, ax) = pyplot.subplots(len(variables), 1, sharex = True)
step = int(1. / dt)
for (i, k) in enumerate(variables):
    ax[i].plot(x, sol[k][ : : step].T)
    ax[i].set_ylabel(k)
ax[-1].set_xlabel('distance (m)')


(fig, ax) = pyplot.subplots(1, 1)
ax.plot(t[ : -1] + numpy.diff(t), speed['P_I'])
ax.axhline(cstar, color = 'black', linestyle = 'dotted')
ax.set_xlabel('time')
ax.set_ylabel('speed')


pyplot.show()
