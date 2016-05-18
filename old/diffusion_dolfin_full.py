#!/usr/bin/python
#

import dolfin
import numpy
from matplotlib import pyplot

x_max = 50.
t_max = 10.

r = t_max
D = t_max / (2. * x_max) ** 2

nx = 100
nt = 100

mesh = dolfin.UnitSquareMesh(nt, nx)
tau = mesh.coordinates()[ : nt + 1, 0]
xi = mesh.coordinates()[ : : nt + 1, 1]

t = t_max * tau
x = x_max * (2. * xi - 1)

# dolfin.set_log_level(dolfin.WARNING)

V = dolfin.FunctionSpace(mesh, 'CG', 1)

# Boundary conditions
def boundary_initial(X, on_boundary):
    tol = 1e-14
    return on_boundary and abs(X[0] - tau[0]) < tol
def boundary_terminal(X, on_boundary):
    tol = 1e-14
    return on_boundary and abs(X[0] - tau[-1]) < tol
def boundary_left(X, on_boundary):
    tol = 1e-14
    return on_boundary and abs(X[1] - xi[0]) < tol
def boundary_right(X, on_boundary):
    tol = 1e-14
    return on_boundary and abs(X[1] - xi[-1]) < tol

# Initial condition
# 1 to the left of xi = 1/2, 0 to the right of xi = 1/2.
# u0 = dolfin.Expression('x[1] < 0.5 ? 1. : 0.')
# Smoother.
u0 = dolfin.Expression('atan(100 * (0.5 - x[1])) / pi + 0.5')
bc_initial = dolfin.DirichletBC(V, dolfin.interpolate(u0, V), boundary_initial)

# With no explicit BC, it's Neumann.
# bc_left = DirichletBC(V, dolfin.Constant(0), boundary_left)
bc_right = dolfin.DirichletBC(V, dolfin.Constant(0), boundary_right)

bcs = [bc_initial, bc_right]


u = dolfin.Function(V)
phi = dolfin.TestFunction(V)

F = ((dolfin.Dx(u, 0)  * phi
      - r * u * (1. - u) * phi
      + D * dolfin.Dx(u, 1) * dolfin.Dx(phi, 1))
     * dolfin.dx)

dolfin.solve(F == 0, u, bcs,
             solver_parameters = {'newton_solver': {'maximum_iterations': 25}})
