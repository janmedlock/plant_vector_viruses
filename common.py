import inspect
import os.path

import matplotlib
import matplotlib.ticker
import numpy
import scipy.special
import seaborn


tmax = 100
t = numpy.linspace(0, tmax, 1001)

vi0 = 0.01

sensitivity_parameters = (
    ('rho', 'Vector birth rate\n$\\rho$ (d$^{-1}$)'),
    ('mu', 'Vector death rate\n$\\mu$ (d$^{-1}$)'),
    ('epsilon', 'Encounter rate\n$\\epsilon$ (d$^{-1}$)'),
    ('phi', 'Proportion feeding\n$\\phi$')
)


seaborn.set_palette('Dark2')
alpha = 0.9

baseline_style = dict(
    linestyle = 'dotted',
    color = 'black',
    linewidth = matplotlib.rcParams['lines.linewidth'] / 2,
    alpha = alpha / 2,
    zorder = 1
)


def get_scale(param):
    if param == 'phi':
        return 'logit'
    else:
        return 'log'


def logitspace(start, stop, *args, **kwargs):
    return scipy.special.expit(numpy.linspace(start, stop, *args, **kwargs))


sensitivity_max_abs_mult_change = 3
npoints = 201


def get_dPs(param, value_baseline):
    scale = get_scale(param)
    if scale in ('linear', 'logit'):
        a = 0.1
        b = 1 - a
        assert a <= value_baseline <= b
        if scale == 'linear':
            return numpy.linspace(a, b, npoints)
        else:
            a, b = scipy.special.logit((a, b))
            return logitspace(a, b, npoints)
    elif scale == 'log':
        a = numpy.log10(value_baseline / sensitivity_max_abs_mult_change)
        b = numpy.log10(value_baseline * sensitivity_max_abs_mult_change)
        return numpy.logspace(a, b, npoints)
    else:
        raise NotImplementedError('scale = {}'.format(scale))


def style_axis(ax):
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))


def _get_filebase():
    stack = inspect.stack()
    caller = stack[2]
    _, basename = os.path.split(caller.filename)
    filebase, _ = os.path.splitext(basename)
    return filebase


def savefig(fig, append = '', format_ = 'pdf', *args, **kwargs):
    filebase = _get_filebase()
    outfile = '{}{}.{}'.format(filebase, append, format_)
    fig.savefig(outfile, *args, **kwargs)


def load_or_build_data(builder):
    filebase = _get_filebase()
    savefile = '{}.npy'.format(filebase)
    try:
        retval = numpy.load(savefile)
    except FileNotFoundError:
        retval = builder()
        numpy.save(savefile, retval)
    return retval
