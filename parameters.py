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
    V0 = 100
    P0 = 10000

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
            # return numpy.nan
            return numpy.inf
        else:
            return self.fV / (1 - self.phiV)

    @property
    def tauV(self):
        if self.phiV == 0:
            # Undefined.
            # return numpy.nan
            return numpy.inf
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


class Persistent(Parameters):
    betaV = 8.3
    betaP = 5.5
    alphaV = 48
    gammaV = 0


class Nonpersistent(Parameters):
    betaV = 500
    betaP = 1000
    alphaV = 86400
    gammaV = 0.05 * 24


parameter_sets = collections.OrderedDict()
parameter_sets['Persistent'] = Persistent()
parameter_sets['Non-persistent'] = Nonpersistent()
