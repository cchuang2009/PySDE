
from pysde import sde
from pysde.sde import *
from sympy import *
#from scitools import *


def __call__(self, *args, **kwargs):
        import inspect
        frame = inspect.currentframe().f_back.f_globals
        nkwargs = {}
        try:
            for k, v in kwargs.items():
                nkwargs[eval(k, frame)] = v
        finally:
            del frame
        return self._subs_dict(nkwargs)

"""
f.subs({x:1})
f.subs(dict(x=1))
f.subs(x=1)
"""


x,dx=symbols('x dx')

ItoInit(x,dx)

r,dr=symbols('r dr')

ItoInit(r,dr)

dw= Symbol('dw')
w = Symbol('w')
BrownSingle(w,dw,0)

"""
print CurrentSMG
print tableIto
print Driftbydt,ItoIntegral
"""
"""
test1=ItoIntdb(w*w)
test2=ItoD(w*w*w)             
print " The Ito integral of w is %s" %(test1)
print " The Ito differential of w*w is %s" %(test2)
"""
#global_assumptions.add(Assume(t, Q.positive))

print(ItoIntdb(exp(w)))
print(ItoIntdb(w))
"""
Mean=simplify(ExpBm(test1))
print " Expectation of int(w,dw) is %s" %(Mean)
"""
[a,b,c,d]=[1,1,1,1]
coeff=[a,b,c,d]
"""
drift=a+b*x
diffusion=c+d*x

drift=1
diffusion=2*sqrt(x)
drift=-x/2
diffusion=sqrt(1-x*x)

drift=-x/2
diffusion=sqrt(1-x*x)

# S(n): make rational power
drift=3*x**(S(1)/3)
diffusion=3*x**(S(2)/3)
"""
drift=a+b*x
diffusion=c+d*x

X0=Symbol('X0')
x0=1.
t0=0
tn=1
sol=SDE_solver(drift,diffusion,t0,x0)
print(" Solution of dX =  (%s) dt + (%s) dw  is %s" %(drift,diffusion,sol))

#print ItoD((w+X0**(S(1)/3))**3)
nt=500
T= linspace(t0, tn, nt+1)
#X=euler(drift,diffusion,x0,t0,tn,nt)
"""
X,Y=milstein(drift,diffusion,x0,t0,tn,nt)

plot(T, X, 'b-4', T, Y, 'r-3',dpi=1200, hardcopy="test.png")
"""
data= fksim_v('1','1','0','1/(1+x*x)')


xx=linspace(-2,2,101)
plot(xx,data[0,:],xx,data[4,:],dpi=1200, hardcopy="test.png")
show()
show()
"""
plot(t, X,'b-4', t, Y, 'r-3', \
         legend=('Euler','Milstein'), \
         dpi=1200, hardcopy="test.png")
"""
#sdeplot(drift,diffusion,1,0,1)
#sdeplot1(coeff,t0,x0,tn)
