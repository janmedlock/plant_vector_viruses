import inspect
import os.path

from matplotlib import pyplot
from matplotlib import ticker
import numpy
import seaborn


tmax = 100
t = numpy.linspace(0, tmax, 1001)

vi0 = 0.01

sensitivity_parameters = (
    ('rho', 'Vector birth rate\n$\\rho$ (d$^{-1}$)'),
    ('mu', 'Vector death rate\n$\\mu$ (d$^{-1}$)'),
    ('phi', 'Time feeding\n$\\phi$'),
    ('epsilon', 'Encounter rate\n$\\epsilon$ (d$^{-1}$)'))


seaborn.set_palette('Dark2')
alpha = 0.9

baseline_style = dict(
    linestyle = 'dotted',
    color = 'black',
    linewidth = pyplot.rcParams['lines.linewidth'] / 2,
    alpha = alpha / 2
)


def get_scale(param):
    if param == 'phi':
        return 'linear'
    else:
        return 'log'


sensitivity_max_abs_mult_change = 4
npoints = 201


def get_dPs(param, value_baseline):
    scale = get_scale(param)
    if scale == 'linear':
        eps = 0.1
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
