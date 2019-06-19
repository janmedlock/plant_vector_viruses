# plant_vector_viruses

## Species interactions affect the spread of vector-borne plant pathogens independent of transmission mode

**[Jan Medlock](http://people.oregonstate.edu/~medlockj/)
[\<jan.medlock@oregonstate.edu\>](mailto:jan.medlock@oregonstate.edu),
Department of Biomedical Sciences, Oregon State University, OR, USA; \
David W. Crowder, Department of Entomology,
Washington State University, WA, USA; \
Jing Li, Department of Mathematics,
California State University Northridge, CA, USA; \
Elizabeth T. Borer, Department of Ecology, Evolution, and Behavior,
University of Minnesota, MN, USA; \
Deborah L. Finke, Division of Plant Sciences,
University of Missouri, MO, USA; \
Rakefet Sharon, MIGAL-Galilee Research Institute,
Northern Research & Development, Israel; \
David Pattemore,
The New Zealand Institute for Plant & Food Research Limited,
Hamilton, NZ.**

**Copyright 2015–2019, Jan Medlock et al.  All rights reserved.
Released under the [GNU AGPL 3](LICENSE).**


This repository contains code used to simulate and
analyze the effects species interactions on the spread of vector-borne
plant pathogens.
> Crowder DW, Li J, Borer ET, Finke DL, Sharon R, Pattemore D,
> Medlock J.
> 2019.
> Species interactions affect the spread of vector-borne plant pathogens
> independent of transmission mode.
> *Ecology*.
> e02782.
> [doi:10.1002/ecy.2782](https://doi.org/10.1002/ecy.2782).

The scripts and model code are written in Python3, using some
third-party libraries.  See
[Python3](https://www.python.org/),
[NumPy & SciPy](https://www.scipy.org/),
[matplotlib](https://matplotlib.org/),
[pandas](https://pandas.pydata.org/),
[Seaborn](https://seaborn.pydata.org/),
[joblib](https://github.com/joblib/joblib/), and
[ad](https://pythonhosted.org/ad/).

### Files

* Core
    * [odes.py](odes.py) defines the model as a system of differential
	  equations and functions for building initial conditions and
	  solving the model.
    * [parameters.py](parameters.py) defines two baseline parameter
      sets, one for a persistent pathogen and one for a non-persistent
      pathogen.
	* [common.py](common.py) has common definitions and functions used
      by the analysis and plotting scripts.
* Analysis & plotting scripts
	* [solutions.py](solutions.py) plots model solutions vs. time for
      the two basic parameter sets.
    * [growth_rates.py](growth_rates.py) analyzes & plots the pathogen
      intrinsic growth rate vs. initial vector population size.
    * [sensitivity_1param.py](sensitivity_1param.py) analyzes & plots
      the sensitivity of the pathogen intrinsic growth rate to the
	  parameters, 1 parameter at a time.
	* [sensitivity_2params.py](sensitivity_2params.py) analyzes & plots
      the sensitivity of the pathogen intrinsic growth rate to the
	  parameters, 2 parameters at a time.
	* [sensitivity_pairs.py](sensitivity_pairs.py) analyzes & plots
      the sensitivity of the pathogen intrinsic growth rate to the
	  parameters, for pairs of parameters that are linked to each
      other.
