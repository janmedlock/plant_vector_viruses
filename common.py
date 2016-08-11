import inspect
import os.path
import warnings

from matplotlib import ticker
import numpy
warnings.filterwarnings(
    'ignore',
    module = 'matplotlib',
    message = ('axes.color_cycle is deprecated '
               'and replaced with axes.prop_cycle; '
               'please use the latter.'))
import seaborn


tmax = 150
t = numpy.linspace(0, tmax, 21)

vi0 = 0.01

sensitivity_parameters = (
    ('bV', 'Vector fecundity\n$b_V$ (d$^{-1}$)'),
    ('muVf', 'Feeding vector mortality\n$\\mu_{Vf}$ (d$^{-1}$)'),
    ('muVm', 'Moving vector mortality\n$\\mu_{Vm}$ (d$^{-1}$)'),
    ('phiV', 'Proportion of vectors feeding\n$\\phi_V$'),
    ('fV', 'Feeding rate\n$f_V$ (d$^{-1}$)'))

sensitivity_max_abs_mult_change = 2
npoints = 21
sensitivity_dPs = numpy.logspace(
    - numpy.log10(sensitivity_max_abs_mult_change),
    + numpy.log10(sensitivity_max_abs_mult_change),
    npoints)


def style_axis(ax):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))


def savefig(fig, append = '', format_ = 'pdf', *args, **kwargs):
    stack = inspect.stack()
    caller = stack[1]
    path, basename = os.path.split(caller.filename)
    filebase, _ = os.path.splitext(basename)
    outfile = os.path.join(path,
                           'Dropbox',
                           '{}{}.{}'.format(filebase, append, format_))
    fig.savefig(outfile, *args, **kwargs)
