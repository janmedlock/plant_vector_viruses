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


# RK4
K1 = {k: dolfin.Function(S) for k in variables}
K2 = {k: dolfin.Function(S) for k in variables}
K3 = {k: dolfin.Function(S) for k in variables}
K4 = {k: dolfin.Function(S) for k in variables}

# K1 = F(Y0)
RHS1 = RHS(Y0)
F_K1 = {k: (K1[k] * Phi[k] - RHS1[k]) * dolfin.dx
        for k in variables}

# K2 = F(Y0 + dt / 2 * K1)
RHS2 = RHS({k: (Y0[k] + dt / 2. * K1[k]) for k in variables})
F_K2 = {k: (K2[k] * Phi[k] - RHS2[k]) * dolfin.dx
        for k in variables}

# K3 = F(Y0 + dt / 2 * K2)
RHS3 = RHS({k: (Y0[k] + dt / 2. * K2[k]) for k in variables})
F_K3 = {k: (K3[k] * Phi[k] - RHS3[k]) * dolfin.dx
        for k in variables}

# K4 = F(Y0 + dt * K3)
RHS4 = RHS({k: Y0[k] + dt * K3[k] for k in variables})
F_K4 = {k: (K4[k] * Phi[k] - RHS4[k]) * dolfin.dx
        for k in variables}

# Y1 = Y0 + dt / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
F_Y1 = {k: ((Y1[k]
             - (Y0[k] + dt / 6. * (K1[k] + 2 * K2[k] + 2 * K3[k] + K4[k])))
            * Phi[k] * dolfin.dx)
        for k in variables}

# Set initial condition.
for k in variables:
    Y0[k].interpolate(IC[k])

sp = {'newton_solver': {'maximum_iterations': 100}}
for i in range(1, len(t)):
    # Find K1.
    for k in variables:
        dolfin.solve(F_K1[k] == 0, K1[k], bcs[k], solver_parameters = sp)
    # Find K2.
    for k in variables:
        dolfin.solve(F_K2[k] == 0, K2[k], bcs[k], solver_parameters = sp)
    # Find K3.
    for k in variables:
        dolfin.solve(F_K3[k] == 0, K3[k], bcs[k], solver_parameters = sp)
    # Find K4.
    for k in variables:
        dolfin.solve(F_K4[k] == 0, K4[k], bcs[k], solver_parameters = sp)
    # Find Y1.
    for k in variables:
        dolfin.solve(F_Y1[k] == 0, Y1[k], bcs[k], solver_parameters = sp)

    # Set Y0 = Y1.
    for k in variables:
        Y0[k].assign(Y1[k])
