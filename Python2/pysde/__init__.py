"""
python SDE -- Python for Stochastic Differential Equations

Try to possibly do symbolic stochastic calculation directory in minimum resources.
  
This package is developed referenced to the following resources:
  
1. Kendall,W.S.: Symbolic Ito calculus in AXIOM: an ongoing story,
   http://www.warwick.ac.uk/statsdept/staff/WSK/abstracts.html#327
2. Sasha Cyganowski: Solving Stochastic Differential Equations with
   Maple, http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.134.7739&rep=rep1&type=pdf
3. Sasha Cyganowski: Maple stochastic package, 
   http://www.math.uni-frankfurt.de/~numerik/maplestoch/

Packages:

sde.py : Main package
test.py: demo file

Necessary Python packages required:

Symbolic:
Sympy: symbolic calculation, http://www.sympy.org

Numerical and simulation plotting:
Numpy: data generating, http://www.scipy.org
scitools: StringFunction for math function and plot for SDE simulation, http://code.google.com/p/scitools/
matplotlib: plot library, http://matplotlib.sf.net

developers:
chu-ching huang,
Math group, Center for General Education,
Chang-Gung University, Taiwan

Email- cchuang@mail.cgu.edu.tw

"""

__version__ = '0.1'
version = __version__
__author__ = 'chu-ching huang'
author = __author__

try:
    from pysde.sde import * 
    from sympy import *
    from scitools import *
    
except ImportError:
    pass
"""
try:
    # for backward compatibility:
    import sys, std
    sys.modules['scitools.all'] = std
    try:
        import TkGUI
        sys.modules['scitools.CanvasCoords'] = TkGUI
        sys.modules['scitools.DrawFunction'] = TkGUI
        sys.modules['scitools.FuncDependenceViz'] = TkGUI
        sys.modules['scitools.FunctionSelector'] = TkGUI
        sys.modules['scitools.ParameterInterface'] = TkGUI
    except ImportError:
        pass  # Pmw and other graphics might be missing - this is not crucial

except ImportError:
    pass

"""

