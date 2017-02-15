import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def check_dependencies():
    install_requires = []

    # Just make sure dependencies exist, I haven't rigorously
    # tested what the minimal versions that will work are
    # (help on that would be awesome)
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import sympy
    except ImportError:
        install_requires.append('sympy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')    
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')

    return install_requires

if __name__ == "__main__":
  
   install_requires = check_dependencies()
   setup(
       name = "pysde",
       version = "1.0.4",
       author = "chu-ching huang, Lars Ericson",
       author_email = "cchuang2009@gmail.com, erxnmedia@hotmail.com",
       description = ("Python Solver via Sympy + SciPy/NumPy for Stochastic Differential Equations!"),
       license = "BSD",
       keywords = "Stochastic differential equations",
       url = "https://github.com/cchuang2009/PySDE",
       packages=['pysde'],
       long_description=read('README.md'),
       classifiers=[
        "Development Status :: 3 - beta",
        "Topic :: Math, Computer",
        "License :: OSI Approved :: BSD License",
       ],
    )
