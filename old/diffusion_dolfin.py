#!/usr/bin/python
#

import dolfin
import numpy
from matplotlib import pyplot

x_max = 50.
t = numpy.linspace(0., 10., 101)
dt = t[1] - t[0]

mesh = dolfin.IntervalMesh(1000, - x_max, x_max)
x = mesh.coordinates()[:, 0]

dolfin.set_log_level(dolfin.WARNING)

V = dolfin.FunctionSpace(mesh, 'CG', 1)

# Boundary conditions
def boundary_left(X, on_boundary):
    tol = 1e-14
    return on_boundary and abs(X[0] - x[0]) < tol
def boundary_right(X, on_boundary):
    tol = 1e-14
    return on_boundary and abs(X[0] - x[-1]) < tol

# With no explicit BC, it's Neumann.
# bc_left = DirichletBC(V, dolfin.Constant(0), boundary_left)
bc_right = dolfin.DirichletBC(V, dolfin.Constant(0), boundary_right)
bcs = [bc_right]

# Initial condition
# 1 to the left of x = 0, 0 to the right of x = 0.
I = dolfin.Expression('x[0] < 0. ? 1. : 0.')
u_1 = dolfin.interpolate(I, V)

u = dolfin.Function(V)
phi = dolfin.TestFunction(V)

F = (((u - u_1 - dt * u_1 * (1 - u)) * phi
      + dt * dolfin.inner(dolfin.nabla_grad(u), dolfin.nabla_grad(phi)))
     * dolfin.dx)

U = numpy.zeros((len(t), mesh.num_cells() + 1))
U[0] = u_1.vector().array()
for i in range(1, len(t)):
    dolfin.solve(F == 0, u, bcs)
    u_1.assign(u)
    U[i] = u_1.vector().array()


threshold = 0.01
position = numpy.zeros(len(t))
for i in range(len(t)):
    position[i] = numpy.interp(- threshold, - U[i], x)
speed = numpy.diff(position) / numpy.diff(t)
cstar = 2


(fig, ax) = pyplot.subplots(2, 1)
step = int(1 / (t[1] - t[0]))
ax[0].plot(x, U[ : : step].T)
ax[0].set_ylabel('density')
ax[0].set_xlabel('distance')

ax[1].plot(t[ : -1] + numpy.diff(t), speed)
ax[1].axhline(cstar, color = 'black', linestyle = 'dotted')
ax[1].set_xlabel('time')
ax[1].set_ylabel('speed')

pyplot.show()
