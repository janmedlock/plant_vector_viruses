import inspect
import os.path

from matplotlib import pyplot
from matplotlib import ticker
import numpy

import seaborn_quiet as seaborn


tmax = 100
t = numpy.linspace(0, tmax, 1001)

vi0 = 0.01

sensitivity_parameters = (
    ('bV', 'Vector fecundity\n$b_V$ (d$^{-1}$)'),
    ('muVf', 'Feeding vector mortality\n$\\mu_{Vf}$ (d$^{-1}$)'),
    ('muVm', 'Moving vector mortality\n$\\mu_{Vm}$ (d$^{-1}$)'),
    ('phiV', 'Proportion of time feeding\n$\\phi_V$'),
    ('fV', 'Encounter rate\n$f_V$ (d$^{-1}$)'))


def get_scale(param):
    if param == 'phiV':
        return 'linear'
    else:
        return 'log'


sensitivity_max_abs_mult_change = 3
npoints = 201


def get_dPs(param, value_baseline):
    scale = get_scale(param)
    if scale == 'linear':
        eps = 0.2
        return numpy.linspace(eps, 1 - eps, npoints)
    elif scale == 'log':
        return value_baseline * numpy.logspace(
            - numpy.log10(sensitivity_max_abs_mult_change),
            + numpy.log10(sensitivity_max_abs_mult_change),
            npoints)
    else:
        raise NotImplementedError('scale = {}'.format(scale))


def style_axis(ax):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))


def savefig(fig, append = '', format_ = 'pdf', *args, **kwargs):
    stack = inspect.stack()
    caller = stack[1]
    _, basename = os.path.split(caller.filename)
    filebase, _ = os.path.splitext(basename)
    outfile = '{}{}.{}'.format(filebase, append, format_)
    fig.savefig(outfile, *args, **kwargs)
