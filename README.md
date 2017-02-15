# Introduction (1.0.5)

<h2>PyS<sup>3</sup>DE= *Py*thon Solver via *S*ympy + *S*ciPy/NumPy for *S*tochastic *D*ifferential *E*quations!</h2> 


**PyS<sup>3</sup>DE** is a solver of stochastic differential equations (SDE) implemented by Python, which both symbolic and numeric schemems are supported.
Numerical solvers include schemes for both w/o jumps.
<br>

<h2>Requirements:</h2>

1. <a href="http://sympy.org">Sympy</br>
2. <a href="http://www.scipy.org">Scipy and Numpy</a></br>
3. <a href="http://matplotlib.org">matplotlib</a>

  
<b>Note:</b> 
1. Scitools is used to generated the picture of simulated data only. 
To keep minimal necessary Python modules/libraries, it would be removed in TODO list. (Thanks, Yoav Ram)
Necessary required submodules of scitools have been added to PySDE since 0.4; this means that scitools is not to be installed.
- after (includeing) 1.0.5, numpy.oldnumeric dependecy was removed ; in otherwords, PySDE works with any version of numpy.
(Thanks to Lars Ericson <erxnmedia@hotmail.com>)
##Optional:

3. <a href="http://code.google.com/p/scitools/">scitools</a><br> 

## Installation

- Extract the source and enter the source directory

   - Python2 version: cd Python2; python setup.py install
   - Python3 version: cd Python3; python setup.py install


## Usages

* Symbolic Computation
<pre>
from sympy import *
from pysde.sde import *
""" Main Codes Here """
x,dx,w,dw,t,dt,a=symbols('x dx w dw t dt a')
x0 =Symbol('x0'); t0 = Symbol('t0')
drift=2\*x/(1+t)-a\*(1+t)\*\*2;diffusion=a\*(1+t)**2
sol=SDE_solver(drift,diffusion,t0,x0)
pprint(sol)  
</pre>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Got


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(<i>t</i>+1)<sup>2</sup> <big><big><big>(</big></big></big>
-<i>a</i><i>t</i>(<i>t</i>₀ + 1)<sup>2</sup>+ <i>a</i> <i>t</i>₀(<i>t</i>₀+ 1)<sup>2</sup>+ <i>a</i><i>w</i>(<i>t</i>₀ + 1)<sup>2</sup>+ <i>x</i>₀<big><big><big>)</big></big></big><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;───────────────────────────<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;
(<i>t</i>₀ + 1)<sup>2</sup>

* Numeric Computation
<pre>
import matplotlib.pylab as plt
from matplotlib import rc
rc('font',\*\*{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
""" setup picture info """
plt.figure(figsize=(5,2))
plt.ylim(-0.5,1.5)
""" Initial data """
x0=1.;t0=0.;tn=10.
x,dx=symbols('x dx')
[a,b,c,d]=[0,-1.,0,1.]
drift=a+b\*x
diffusion=c+d\*x#
nt=200
T= linspace(t0, tn, nt+1)
""" Numerical Computation"""
X=Euler(drift,diffusion,x0,t0,tn,nt)
X,Y=Milstein(drift,diffusion,x0,t0,tn,nt)
"""Make picture"""
plt.plot(T, X, color="blue", linewidth=2.5, linestyle="-", label="Euler")
plt.plot(T, Y, color="red", linewidth=2.5, linestyle="--", label="Milstein")
plt.plot(T, np.exp(-T), color="green", linewidth=2.5, linestyle="--", label=r"$\exp(-t)$")
plt.ylim(X.min()-0.2, X.max()+0.2)
plt.title(r"$d X_t=-dt+d W_t,X_0=1$")
plt.legend()
plt.savefig('Milstein.eps')
</pre>

## Note

* Symbolic/Numberic SDE solvers depend on part of the Scitools module which had been extracted and incorporated with the library, 
  which it is not necessary to install scitools again. (removed since 0.4)
 
* Schemes and examples for simulating SDE’s with jumps adds, see demo/sdedemo.ipynb.

## DEMO

1. python code: demo/demo.py)
- TeXmacs with Python plugin: demo/sdedemo.tm, demo/sdeJump.tm
- Sage notebook(): demo/DiffusionJumps.sws, demo/DiffusionJumps-1.sws
- Jupyter/notebook: demo/sdedemo.ipynb

## Developer:

1. chu-ching huang: cchuang2009@gmail.com
- Lars Ericson: <erxnmedia@hotmail.com>
