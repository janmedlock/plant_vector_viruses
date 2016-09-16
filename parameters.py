#!/usr/bin/python3

import collections
import re
import textwrap

import numpy


class Parameters:
    fV = 6
    phiV = 0.5
    muVf = 0.02
    muVm = 0.04
    bV = 0.08
    KV = 100
    gammaP = 0
    V0 = 100
    P0 = 10000

    def __init__(self):
        self.QSSA = QSSA(self)

    @property
    def muV(self):
        return self.muVf

    @property
    def deltaV(self):
        return self.muVm / self.muVf - 1

    @property
    def sigmaV(self):
        if self.phiV == 1:
            # Undefined.
            return numpy.nan
        else:
            return self.fV / (1 - self.phiV)

    @property
    def tauV(self):
        if self.phiV == 0:
            # Undefined.
            return numpy.nan
        else:
            return self.fV / self.phiV

    def __repr__(self):
        # Start with '<ClassName'
        s = '<{}'.format(self.__class__.__name__)
        params_strs = []
        for a in sorted(dir(self)):
            if not a.startswith('_'):
                v = getattr(self, a)
                if not callable(v):
                    params_strs.append('{} = {}'.format(a, v))
        if len(params_strs) > 0:
            s += ': '
            width = len(s)  # of '<ClassName: '
            s += ', '.join(params_strs)
        else:
            width = 1  # of '<'
        s += '>'
        # Wrap on commas and indent.
        wrapper = textwrap.TextWrapper(subsequent_indent = ' ' * width)
        nonspace = '*'  # A char that textwrap.TextWrapper won't linebreak on.
        # Replace spaces not preceded by commas with non-spaces.
        s_nonspaced = re.sub(r'(?<!,) ', nonspace, s)
        s_nonspaced_wrapped = wrapper.fill(s_nonspaced)
        # Replace non-spaces with spaces.
        s_wrapped = s_nonspaced_wrapped.replace(nonspace, ' ')
        return s_wrapped


class QSSA:
    def __init__(self, params):
        self.params = params

    @property
    def vf(self):
        return self.params.phiV

    @property
    def vm(self):
        return 1 - self.params.phiV

    @property
    def muV(self):
        muV_ = self.params.muV * (1 + self.params.deltaV * self.vm)
        return muV_

    @property
    def bV(self):
        bV_ = self.vf * self.params.bV + self.vm * 0
        return bV_

    @property
    def Vinf(self):
        Vinf_ = ((self.bV - self.muV) / self.bV
                 * self.params.KV * self.params.P0)
        return Vinf_

    def V(self, t):
        if (self.bV != self.muV):
            V_ = (self.params.V0 * numpy.exp((self.bV - self.muV) * t)
                  / (1 + self.params.V0 / self.Vinf
                     * (numpy.exp((self.bV - self.muV) * t) - 1)))
        else:
            V_ = (self.params.V0
                  / (1
                     + self.bV * self.params.V0 / self.params.KV
                     / self.params.P0 * t))
        return V_

    def r0(self, t):
        r0_ = (- (self.muV + self.params.gammaV + self.params.gammaP)
               + numpy.sqrt(
                   (self.muV + self.params.gammaV + self.params.gammaP) ** 2
                   + 4 * (self.params.betaV * self.params.betaP * self.vf ** 2
                          * self.V(t) / self.params.P0
                          - self.params.gammaP
                          * (self.muV + self.params.gammaV)))) / 2
        return r0_


class Persistent(Parameters):
    betaV = 0.02 * 24
    betaP = 0.02 * 24
    gammaV = 0


class Nonpersistent(Parameters):
    betaV = 0.2 * 24
    betaP = 0.2 * 24
    gammaV = 0.05 * 24


parameter_sets = collections.OrderedDict()
parameter_sets['Persistent'] = Persistent()
parameter_sets['Non-persistent'] = Nonpersistent()
