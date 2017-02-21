#!/usr/bin/python3

import collections


class Parameters:
    epsilon = 3    # Feeding on 3 plant per day
    phi = 0.5      # 50% of vectors' time is spent feeding
    mu_f = 0.02    # 2% mortality per day while feeding
    mu_m = 0.04    # 4% mortality per day while moving
    rho = 0.08     # 8% birth rate per day
    kappa = 100    # 100 feeding vectors per plant carrying capacity
    V0 = 100       # 100 initial vectors
    P0 = 10000     # 10000 plants

    @property
    def mu(self):
        return self.phi * self.mu_f + (1 - self.phi) * self.mu_m

    @mu.setter
    def mu(self, mu_new):
        mu_old = self.mu
        self.mu_f *= mu_new / mu_old
        self.mu_m *= mu_new / mu_old


class Persistent(Parameters):
    beta_V = 8.3    # 50% acquisition at 2 hrs
    beta_P = 5.5    # 50% innoculation at 3 hrs
    alpha = 48     # delay to acquisition/innoculation of 30 mins
    gamma_f = 0    # no clearance
    gamma_m = 0    # no clearance


class Nonpersistent(Parameters):
    beta_V = 500    # 50% acquisition at 2 mins.
    beta_P = 10000  # 50% innoculation at 0.1 mins.
    alpha = 86400  # delay to acquisition/innoculation of 1 sec
    gamma_f = 288  # 5 mins to clearance while feeding
    gamma_m = 24   # 1 hour to 50% clearance while moving


parameter_sets = collections.OrderedDict()
parameter_sets['Persistent'] = Persistent()
parameter_sets['Non-persistent'] = Nonpersistent()
