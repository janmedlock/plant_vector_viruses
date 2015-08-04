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


S0 = dolfin.FunctionSpace(mesh, 'DG', 1)
S = dolfin.MixedFunctionSpace((S0, ) * 3)


dolfin.set_log_level(dolfin.INFO)


# Boundary conditions
def boundary_right(X, on_boundary):
    tol = 1e-14
    return on_boundary and abs(X[0] - x[-1]) < tol

# Same BCs for all components.
# Neumann on the left: no explicit BC there.
bcs_right = dolfin.DirichletBC(S,
                               dolfin.Constant((0., 0., 0.)),
                               boundary_right)
bcs = bcs_right


# Initial condition
# Right-moving wave.
Vstar = K_V * (1. - mu_V / b_V)
V_Istar = Vstar * beta_V / (mu_V + beta_V)
IC = dolfin.Expression(['x[0] < 0. ? {} : {}'.format(Vstar - V_Istar,  Vstar),
                        'x[0] < 0. ? {} : {}'.format(V_Istar, 0.),
                        'x[0] < 0. ? {} : {}'.format(P, 0.)])


Y0 = dolfin.Function(S)
Y1 = dolfin.Function(S)
Phi = dolfin.TestFunction(S)


def Dxx(y, phi):
    return - dolfin.inner(dolfin.nabla_grad(y), dolfin.nabla_grad(phi))

def RHS(Y):
    (V_S, V_I, P_I) = dolfin.split(Y)
    (phiV_S, phiV_I, phiP_I) = dolfin.split(Phi)

    dV_S = ((b_V * (V_S + V_I) * (1 - (V_S + V_I) / K_V)  # birth
             - mu_V * V_S                                 # death
             - beta_V * P_I / P * V_S) * phiV_S           # infection
            + D_V * Dxx(V_S, phiV_S))                     # diffusion

    dV_I = ((- mu_V * V_I                                 # death
             + beta_V * P_I / P * V_S) * phiV_I           # infection
            + D_V * Dxx(V_S, phiV_I))                     # diffusion

    dP_I = beta_P * V_I * (1. - P_I / P) * phiP_I         # infection

    return dV_S + dV_I + dP_I


# RK4
# K1 = F(Y0)
K1 = dolfin.Function(S)
F_K1 = (dolfin.inner(K1, Phi) - RHS(Y0)) * dolfin.dx

# Y01 = Y0 + dt / 2 * K1
Y01 = dolfin.Function(S)
F_Y011 = (dolfin.inner(Y01, Phi)
          - (dolfin.inner(Y0, Phi)
             + dt / 2. * dolfin.inner(K1, Phi))) * dolfin.dx
# K2 = F(Y01)
K2 = dolfin.Function(S)
F_K2 = (dolfin.inner(K2, Phi) - RHS(Y01)) * dolfin.dx

# Y01 = Y0 + dt / 2 * K2
F_Y012 = (dolfin.inner(Y01, Phi)
          - (dolfin.inner(Y0, Phi)
             + dt / 2. * dolfin.inner(K2, Phi))) * dolfin.dx
# K3 = F(Y02)
K3 = dolfin.Function(S)
F_K3 = (dolfin.inner(K3, Phi) - RHS(Y01)) * dolfin.dx

# Y01 = Y0 + dt * K3
F_Y013 = (dolfin.inner(Y01, Phi)
          - (dolfin.inner(Y0, Phi)
             + dt * dolfin.inner(K3, Phi))) * dolfin.dx
# K4 = F(Y0 + dt * K3)
K4 = dolfin.Function(S)
F_K4 = (dolfin.inner(K4, Phi) - RHS(Y01)) * dolfin.dx

# Y1 = Y0 + dt / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
F_Y1 = (dolfin.inner(Y1, Phi)
        - (dolfin.inner(Y0, Phi)
           + dt / 6. * (dolfin.inner(K1, Phi)
                        + 2. * dolfin.inner(K2, Phi)
                        + 2. * dolfin.inner(K3, Phi)
                        + dolfin.inner(K4, Phi)))) * dolfin.dx


# Set initial condition.
Y0.interpolate(IC)

sp = {'newton_solver': {'maximum_iterations': 25}}
sol= numpy.zeros((S.num_sub_spaces(), len(t), mesh.num_cells() + 1))
for i in range(S.num_sub_spaces()):
    sol[i][0] = dolfin.project(Y0.sub(i)).vector().array()
for i in range(1, len(t)):
    # Find K1.
    dolfin.solve(F_K1 == 0, K1, bcs, solver_parameters = sp)
    # Find K2.
    dolfin.solve(F_Y011 == 0, Y01, bcs, solver_parameters = sp)
    dolfin.solve(F_K2 == 0, K2, bcs, solver_parameters = sp)
    # Find K3.
    dolfin.solve(F_Y012 == 0, Y01, bcs, solver_parameters = sp)
    dolfin.solve(F_K3 == 0, K3, bcs, solver_parameters = sp)
    # Find K4.
    dolfin.solve(F_Y013 == 0, Y01, bcs, solver_parameters = sp)
    dolfin.solve(F_K4 == 0, K4, bcs, solver_parameters = sp)
    # Find Y1.
    dolfin.solve(F_Y1 == 0, Y1, bcs, solver_parameters = sp)

    # Set Y0 = Y1 and store.
    Y0.assign(Y1)
    for j in range(S.num_sub_spaces()):
        sol[j][i] = dolfin.project(Y0.sub(j)).vector().array()


threshold = 0.01
position = numpy.zeros((S.num_sub_spaces(), len(t)))
for i in range(len(t)):
    for j in range(S.num_sub_spaces()):
        position[j][i] = numpy.interp(- threshold, - sol[j][i], x)
speed = numpy.diff(position, axis = 1) / numpy.diff(t)


q0 = beta_P * beta_V * Vstar / P
q1 = mu_V ** 2 + 3 * q0
q2 = 2 * mu_V ** 2 + 9 * q0
q3 = mu_V ** 2 + 4 * q0
cstar = numpy.sqrt(D_V * (2 * q1 ** (3. / 2) - mu_V * q2) / q3)

(fig, ax) = pyplot.subplots(S.num_sub_spaces(), 1, sharex = True)
step = int(1. / dt)
for i in range(S.num_sub_spaces()):
    ax[i].plot(x, sol[i][ : : step].T)
    # ax[i].set_ylabel(i)
ax[-1].set_xlabel('distance (m)')


(fig, ax) = pyplot.subplots(1, 1)
ax.plot(t[ : -1] + numpy.diff(t), speed[2])
ax.axhline(cstar, color = 'black', linestyle = 'dotted')
ax.set_xlabel('time')
ax.set_ylabel('speed')


pyplot.show()
