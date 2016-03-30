#!/usr/bin/python

import numpy
from matplotlib import pyplot
from scipy import integrate


b_V = 1.
mu_V = 0.05
gamma_V = 0

def elasticity_R02(elasticity_b, elasticity_mu):
    return mu_V / (b_V - mu_V) * elasticity_b - mu_V / (mu_V + gamma_V) * elasticity_mu

e_b = e_mu = numpy.linspace(-1, 1, 51)

E_b, E_mu = numpy.meshgrid(e_b, e_mu)

E_R02 = elasticity_R02(E_b, E_mu)


vabsmax = numpy.abs(E_R02).max()
# pyplot.imshow(E_R02, aspect = 'equal', interpolation = 'nearest', origin = 'lower',
#               cmap = 'RdBu', vmin = -vabsmax, vmax = vabsmax,
#               extent = (e_b.min(), e_b.max(), e_mu.min(), e_mu.max()))
pyplot.pcolor(E_b, E_mu, E_R02,
              cmap = 'RdBu', vmin = -vabsmax, vmax = vabsmax)

pyplot.xlabel('$\epsilon_{b_V}$')
pyplot.ylabel('$\epsilon_{\mu_V}$')
pyplot.colorbar()


# sigma_V = 0.1
# tau_V = 1.

# t = numpy.linspace(0., 5. * max(1. / tau_V, 1. / sigma_V), 101)

# V_F = numpy.exp(- tau_V * t)

# V_W = tau_V / (sigma_V - tau_V) * (numpy.exp(- tau_V * t)
#                                    - numpy.exp(- sigma_V * t))
# V_W = tau_V / (tau_V - sigma_V) * (numpy.exp(- sigma_V * t)
#                                    - numpy.exp(- tau_V * t))

# V_G = (1
#        - tau_V / (tau_V - sigma_V) * numpy.exp(- sigma_V * t)
#        + sigma_V / (tau_V - sigma_V) * numpy.exp(- tau_V * t))

# pyplot.plot(t, V_G)



pyplot.show()
