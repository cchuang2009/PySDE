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
matplotlib: plot library, http://matplotlib.sf.net

developers:
chu-ching huang,
Math group, Center for General Education,
Chang-Gung University, Taiwan

Email- cchuang@mail.cgu.edu.tw

"""

import copy
from copy import copy
from numpy import *
import numpy as np
from pysde.StringFunction import StringFunction
from sympy import *
from IPython.display import display, Math, Latex

def show(sol):
    display(Math(latex(sol)))

CurrentDiff=[]
CurrentSMG=[]
Driftbydt=[]

tableBSD = {}
tableIto = {}

Driftbydt={}
ItoIntegral={}
Fixed={}

t, dt=symbols('t dt')
Driftbydt={}

def make_Func(t,*args):
    """
    create function of t
    Usage:
    f,g,...=make_Func('f','g',...)
    return
    f(t),g(t), etc
    """
    return [Function(s)(t) for s in args]

def ItoD(f):
    """
    ItoD(f)= $d (f(t,w))$
    """
    w,dw = symbols('w dw')
    ItoDiff= diff(f,w)*dw+diff(f,w,2)/2*dt+diff(f,t)*dt
    return ItoDiff

def ItoIntdb(f):
    """
    ItoIntdb(f) = $\int f(t,w) dw $
    """
    s, t, dt, w, dw, x= symbols('s t dt w dw x')
    if diff(f,t)==0 and diff(f,w)==0:
       """
       $\int a dw = a w$
       """
       return f*w
    elif diff(f,t)!=0 and diff(f,w)==0:
       """
       $\int f(t) d w = N(0, \int f^2 dt)$
       """
       stdvar = integrate(f*f,t)
       #stdvar=Function("N")(t)
       result=N(0,stdvar)
       return result.rv()
    elif diff(f,w)!=0 and diff(f,t)==0:
       """
       $\int f(w) d w = F(w)-F(0)-1/2*\int f'(w) dt$
       """
       F = integrate(f,w)
       fp = Function('fp')(t)
       fp = diff(f,w)
       if diff(fp,w)==0.:
           ItoInt=F-F.subs({w:0})-fp*t/2
       else:    
           ItoInt = F-F.subs({w:0})-Integral(fp,t)/2
       return ItoInt
    else:
       F = integrate(f,w)
       ft = Function('ft')([t])
       fw = Function('fw')([t])
       fw = diff(f,w)
       Ft = diff(F,t)
       ItoInt = F-F.subs({w:0,t:0})-Integral(Ft,t)-Integral(fw,t)/2
       return ItoInt
class N(Function):
    def __init__(self,mean,variance):
        self=Function('N')(t)
        self.mean=mean;
        self.variance=variance;
    def m(self):
        return self.mean
    def var(self):
        return self.variance
    def rv(self):
        #return "N(%s,%s)" %(self.mean,self.variance)
        return self

def ItoIntft(f,t0=0,tn=t):
    if diff(f,t)!=0:
       f=f.subs(dict(t=s))
       v=integrate(f*f,(s,t0,tn))
       result=N(0,v)
       return result.rv()   
    else:
       return f*w
    
"""
def ItoIntf(f):
    s, t, dt, w,dw = symbols('s t dt w dw')
    if diff(f,t)==0:
        return f*w
    else:
       #f.subs({t:s})
       F = integrate(f*f,t)
       N=Function("N")(t)
       return "N(0.%s)" %(F)
"""

def ExpBm(f):
    w,t,b = symbols('w t b')
    global_assumptions.add(Assume(t, Q.positive))
    pdf1=exp(-b*b/2)/sqrt(2*pi)
    fb=f.subs({w:b*sqrt(t)})
    integrand=pdf1*fb
    return integrate(integrand,(b,-oo,oo))
    
def ItoExpand(sd):
    return

def ItoDrift(sd):
    return

def ItoIntroduce(x_ito,dx_ito):
    if len(CurrentSMG)==0:
       CurrentDiff.append(dx_ito)
       CurrentSMG.append(x_ito)
       tableIto[x_ito]=dx_ito
       tableBSD[dx_ito]=x_ito
    elif (x_ito not in CurrentSMG):
       CurrentDiff.append(dx_ito)
       CurrentSMG.append(x_ito)
       tableIto[x_ito]=dx_ito
       tableBSD[dx_ito]=x_ito
    return
    
def AddQuadVar(*args):
    """
    if len(args)==3:
        dx=arg3[0];dy=args[1];xdt=args[2];
    elseif
    """
    return

def AddDrift(dx, driftdx):
    Driftbydt[dx]=driftdx/dt
    return

def AddFixed(t0, y, y0):
    return

def InitialValue(t0, x):
    return

def ItoInit(tvar,dtvar):
    t=tvar;dt=dtvar
    t0=Symbol('t0')
    myrules=[]

    ItoIntroduce(t,dt)
    AddDrift(dt,dt)
    ItoIntegral[dtvar] = tvar
    Fixed[t0,t]=t0
    return

def BrownSingle(x,dx,x0):
    ItoIntroduce(x,dx)
    AddQuadVar(dx**2,t)
    AddDrift(dx,0)
    Fixed[0,dx]=x0
    ItoIntegral[dx]=x+x0
    return

def BrownBasis(SMList, IVList):
    return

def BrownPairs(SL):
    return
    
def Itosde(x,eq,x0):
    return

def DiffusionCoefficients(x,dtau):
    return

def ItoStatus():
    return

def ItoClear(x, dx):
    return

def ItoReset(t, dt):
    return

def LSDE_sol(drift,diffusion,t0=0,X0=0):
    """
    Solve
    drift = a X + b
    diffusion = c X +d
    dX = (drift) dt + (diffusion) dW
    X = g(t0,t)(X0+int(g^(-1)(t0,s)(b-cd),ds,t0,t)
       + int(g^(-1)d dW)
    g(t0,t)=exp(int(a-c^2/2,ds,t0,t)+int(c,dw,t0,t)
    ...............................................
    
    drift: aX+b
    diffusion: cX+d
    to: initial time
    X0: initial Xt

    General solution for
    $d X = (a X+b) d t + (c X + s) d W$
    is

    $ X_t = M_t^{-1}{X_0 + (a-cd) \int_0^t M_s d s+ d \int_0^tM_s d w_s$
    where
    $ M = exp( (-b+c^2/2)t + c^2 w_t)$
    """
    x,s,t,w = symbols('x s t w')
    
    if (diff(drift,x,2)!=0) or (diff(diffusion,x,2)!=0):
        print("Not linear SDE!")
        exit
    else:
        alpha=diff(drift,x)
        if alpha!=0:
           alpha=alpha.subs({t:s})
           gamma=drift.subs({x:0})
           gamma = gamma.subs({t:s})
        elif diff(drift,t)!=0:
            gamma = drift.subs({t:s})
        else:
           gamma = drift
        
        beta=diff(diffusion,x)
        if beta!=0:
           beta=beta.subs({t:s})
           delta = diffusion.subs({x:0})
           delta = delta.subs({t:s})
        elif diff(diffusion,t)!=0:
           delta = diffusion.subs({t:s})
        else:
           delta = diffusion
        
        if (diff(beta,s)==0):
            temp1=beta*w
        else:
            temp1=ItoIntdb(beta)

        """
        g : integral factor
          = exp( \int(a-b^2/2,(s,t0,t)) + \ItoInt(b,(w_s,t0,t)) )
        g2 = 1/g  
        """
        g=exp(integrate(alpha-1/2.*(beta**2),(s,t0,t))+temp1)
        g2=g.subs({t:s})
        
        if beta == 0:
           I1 = integrate(1/g2*(gamma-beta*delta),(s,t0,t))
           if diff(delta,s)!=0:
              I2 = ItoIntdb(1/g*(delta.subs({s:t})))
           else:
              I2 = ItoIntdb(1/g*delta)
           soln = g*(X0+I1+I2)
        else:
           I1 =  integrate(1/g2*(gamma-beta*delta),(s,t0,t))
           if diff(delta,s)!=0:
              I2 = ItoIntdb(1/g*(delta.subs({s:t})))
           else:
              I2 = ItoIntdb(1/g*delta)
           #I2 =  integrate(1/g2*delta,(w,t0,t))
           #print g,delta
           soln = g*(X0+I1+I2)
    return soln

def RSDE_sol(drift,diffusion,t0=0,X0=0):
    x,s,t,w = symbols('x s t w')
    soln=[]
    factor = integrate(1/diffusion,x)
    alpha1= drift-diffusion*diff(diffusion,x)/2
    alpha=alpha1/diffusion/factor
    alpha12= alpha1.subs({x:1})
    if diff(alpha,x)==0.0:# or alpha12==0.0:
       test=drift-diffusion*diff(diffusion,x)/2
       if test==0:# or alpha12==0.0:
           eqn= factor-(factor.subs({x:X0})+w)
           print(eqn)
           soln=solve(eqn,x)
    else:
           Iw=ItoIntf(alpha*t)
           eqn=-factor-(exp(alpha*t)*factor.subs({x:X0})+exp(alpha*t)*Iw)
           soln=solve(simplify(eqn),x)
    if size(soln)!=0:
       return soln[0]
    else:
       print(" SDE solving Not Implemented!")   

def Reduce2(mu,sigma,t0=0,x0=0):
    """
    Type: 

       $d X_t = \gamma \left( t, X_t \right) d t + b \left( t \right) X_t d W_t$:

    Define

    \[
   
       Y_t \left( \omega \right) = F_t \left( \omega \right) X_t \left( \omega \right)

    \]

    where integrating factor

    \[

       F_t = \exp \left( - \int^t_0 b \left( s \right) d W_t + \frac{1}{2} \int^t_0 b^2 \left( s \right) d s \right)
    \]
    Solution:
        $X_t = F^{- 1}_t Y_t$
    where
    \[
      \begin{matrix}
           \frac{d Y_t \left( \omega \right)}{d t} &= &F_t \left( \omega \right) \cdot 
                 \gamma \left( t, F^{- 1}_t \left(      \omega \right) Y_t \left( \omega \right) \right),  \\

           Y_0 &=& x

      \end{matrix}

    \] 
    """ 
    W =Symbol("W")
    X = Function("X")(W)
    a=Symbol("a")
    r=mu
    b=sigma/X
    F=exp(-integrate(b,W)+integrate(b*b,t)/2)
    Y = Function("Y")(t)
    WW=Function("WW")(t)
    rr=r.subs({X:Y/F})
    Y_prime=Derivative(Y, t)  
    rrr=(rr*F).subs({W:WW}) 
    sol_Y=dsolve(Y_prime-rrr, Y)
    #print sol_Y[0].rhs
    sol=sol_Y[0].rhs.subs({WW:W})/F
    return sol


def SDE_solver(a,b,t0=0,X0=0):
    x = Symbol('x')
    if diff(a,x,2)==0 and diff(b,x,2)==0:
       sol=LSDE_sol(a,b,t0,X0)
    else:
        sol=RSDE_sol(a,b,t0,X0)   
    return sol
    
def L0(X,a,b):
    x,t =symbols('x t')
    Lt=diff(X,t)
    Lx=a*diff(X,x)+b*bdiff(X,x,2)/2.
    Sol=Lt+Lx
    return Sol

def L1(X,b):
    x = Symbol('x')
    return b*diff(X,x)

"""
def pdf_Wright(mu,sigma,a=-oo,b=oo):

    x = Symbol('x')
    P=exp(integrate(mu/sigma,x))/sigma
    phi=1/integrate(P,(x,a,b))
    sol=phi*P  
    return sol
"""

def normal_int(func,var,a=-oo,b=oo):
    """
    normal_int(func,a,b)
     \int_a^b exp(func) dx
    i) \int^\infty_{-infty} (-a x^2+b x+c) dx
      = \sqrt{2\pi/A} exp(b^2/{4a}+c)
      
    y=var
    y= Symbol("y")
    func=func.subs({var:y})
    A=-diff(func,y,2)/2
    B=diff(func,y).subs({y:0})
    C=func.subs({y:0})
    if diff(func,y,3)==0 and a==-oo :
    """

    A=-diff(func,var,2)/2
    B=diff(func,var).subs({var:0})
    C=func.subs({var:0})
    if diff(func,var,3)==0 and a==-oo :
       sol= sqrt(pi/A)*exp(B*B/4/A+C)
    else:
        print("1")
    return simplify(sol)
    
def KolmogorovFE_Spdf(mu,sigma2,a=-oo,b=oo):
    """
Kolmogorov Forward Equation:

\[

\frac{\partial f}{\partial t} = \frac{\partial \mu(x)f(t,x)}{\partial x} +\frac{\partial^2 (\sigma^2(x)f(x,t))}{\partial x^2}

\]

(Wright's Formula: Stationary solution)

\[ f(x)=\frac{\phi}{\sigma^2}\cdot exp\left({\int^x\frac{\mu(s)}{\sigma^2(s)}d s}\right) \]

where $\phi$ is chosen so as to make $\int^{\infty}_{-\infty}f(x) d x=1$
    
    """
    x = Symbol('x')
    expf=mu/sigma2
    m=integrate(expf,x)
    
    if a==-oo and diff(sigma2,x)==0:
       """
        Normal Case:
        $mu=A x+B$
        $sigma^2=C $   
       """ 
       temp=normal_int(m,x,a,b)
       phi=1/temp*sigma2
       sol=simplify(phi*exp(m)/sigma2)
    elif a==0 and  diff(sigma2,x,2)==0:
         if diff(sigma2,x)!=0 and diff(mu,x,2)==0:
            """
            Gamma case:
            $mu=A x+B$
            $sigma^2=C x$
            """
            A=diff(mu,x)
            B=mu.subs({x:0})
            C=diff(sigma2,x)
            phi=C*(-C/A)**B/gamma(B)
            sol=phi/C*x**(B-1)*exp(A*x/C)
    elif a==0 and b==1:
        """
            Beta case:
            $mu=A x+B$
            $sigma^2=C x(1-x)$
        """
        A=diff(mu,x)
        B=mu.subs({x:0})
        C=-diff(sigma2,x,2)/2
        phi=C/gamma(B/C)/gamma(-(1+A/B)*B/C)*gamma(-A/C)
        sol=phi*x**(B/C-1)*(1-x)**(-B/C-A/C-1)/C
    else:
        """
        General Case:
          $\phi*exp(\int^x \mu/\sigma^2 d x)/sigma^2$      
        """
        temp=exp(m)/sigma2       
        phi=1/integrate(temp,(x,a,b))
        sol=simplify(phi*temp)
    return sol   


def Euler(a,b,x0=1,t0=0,tn=1,n=200):
    x,t =symbols('x t')
    T=tn-t0
    nt=n
    dt = T/nt
    sqrtdt = sqrt(dt)
    #t=array([])
    t = linspace(t0, tn, nt+1)    # 101 points between 0 and 3
    #e=ones([1,size(t)])
    
    X=0.*copy(t)
    #y=0.*copy(t)
    dw=0.*copy(t)
    
    X[0]=x0
    #y[0]=x0
    for j in range(0, nt):
        #dw.append(random.gauss(0, sqrtdt))
        #dw[j]= gauss(0, sqrtdt)
        dw[j]=random.randn()*sqrtdt
        #y[j+1]= y[j]+(A+B*y[j])*dt
        if diff(a,x)==0:
           ax=a
        else:
           ax=a.subs({x:X[j]})
        if diff(b,x)==0:
           bx=b
        else:
           bx=b.subs({x:X[j]})   
        X[j+1]= X[j]+ax*dt+bx*dw[j]
    return X
    
def Milstein(a,b,x0=1,t0=0,tn=1,n=200):
    x,t =symbols('x t')
    T=tn-t0
    nt=n
    dt = T/nt
    sqrtdt = sqrt(dt)
    #t=array([])
    t = linspace(t0, tn, nt+1)    # 101 points between 0 and 3
    #e=ones([1,size(t)])
    
    X=0.*copy(t)
    Y=0.*copy(t)
    dw=0.*copy(t)
    
    X[0]=x0
    Y[0]=x0
    for j in range(0, nt):
        #dw.append(random.gauss(0, sqrtdt))
        #dw[j]= gauss(0, sqrtdt)
        dw[j]=random.randn()*sqrtdt
        #y[j+1]= y[j]+(A+B*y[j])*dt
        if diff(a,x)==0:
           ax=a
        else:
           ax=a.subs({x:X[j]})
        if diff(b,x)==0:
           bx=b
        else:
           bx=b.subs({x:X[j]})   
        b_prime=diff(b,x)
        if diff(b_prime,x)==0:
           bp=b_prime
        else:
           bp=b_prime.subs({x:X[j]}) 
        Y[j+1]= Y[j]+ax*dt+bx*dw[j]
        X[j+1]= X[j]+ax*dt+bx*dw[j]+bx*bp*(dw[j]*dw[j]-dt)/2.
    return X,Y

def WPath(T=1.,n=100):
    w=0.;t=0.;h=T/n;
    W=[[0,w]]
    for i in range(0,n):
        t+=h;
        w+=random.normal(0,sqrt(h))
        W.append([t,w])
    return W
  
def Wt(W,t,T=1,n=100):
    """
        interpolation between time [i,i+1] for diffusion with jump
    """
    i=int(n*t/T);
    
    if i==n-1:
       Wi=W[n][1];
    else:   
       dt=t*n/T-i;
       Wi=dt*W[i-1][1]+(1.-dt)*W[i][1]
    return Wi  

def jump(lam=1.,T=1.,mu=0.,sigma2=1.):
    """  
    Despcription:
      jump returns
      U=exp(Gaussian(mu,sigma2))-1
    Syntax:
      > j=jump()
    """
    loop=True;

    JumpU=[];
    
    t=[];
    tjump=0;

    while loop==True:
      tau = random.exponential(lam);
      if tjump+tau<=T:
         tjump+=tau;
         t.append(tjump)
      else:
         loop=False;
    
    #number of jumps	 
    i=size(t);
    #print i, tau,t
    
    
    """
    Simulate the type and magnitutd jump at each t_tau
    """
    for j in range(0,i):
          if sigma2==0:
             U=1;
          else:
             U=exp(random.normal(mu,sqrt(sigma2)))-1
          if j==1:
             JumpU.append([t[j],U]);
          else:
             JumpU.append([t[j],U]);
    if i==0:
       return [];
    else:
       return JumpU

def jumpSum(U=[],t=1.,gamm=1.):
    """
    Estimate the magnitude between [0,t] with parameter, gamma:
    
    \sum \log( 1 + \gamma U[:] )
    
    """

    SumJ=0.
    lam=1;
    
    """
    noJ: number of jumps
    SumS: total magnitude
    """
    noJ=sum(item[0]<=t for item in U)
    SumJ=sum(log(1+gamm*U[k][1]) for k in range(noJ))
   
    return SumJ
    
def dJump(t0=1,t1=1.,U=[]):
    """
    Calculate sum of the total jumps, delta N, between [t0,t1]
    """

    n0=sum(item[0]<=t0 for item in U)
    n1=sum(item[0]<=t1 for item in U)
    S1=sum(U[i][1] for i in range(n0))
    S2=sum(U[j][1] for j in range(n1))
    return S2-S1

def EulerScheme(a,b,c,y,dt,dW,dN):
    return y+a*dt+b*dW+c*dN

    
def bc(a,x,y,c):
    return a.subs({x:y+c})

def AdaptMilsteinScheme(a,b,c,y,dt,dW,dN):
    x=Symbols('x');
    bc=b.subs({x:y+c})
    return y+(a-b*diff(b,x)/2.)*dt+b*dW+b*diff(b,x)*dW**2./2+c*dN+(bc-b)*dW*dN
    
    
def fksim(a,b,c,h,x0=-2.,xn=2.,t0=0.,tn=1.):
    """
    Feynman-Kac formula:
    PDE:
        $$ \frac{d\phi}{d t} = a(x) \frac{d\phi}{d x}+\frac{b^2}{2}\frac{d^2\phi}{d x^2}+c\phi$$
    IC:
        $$\phi(0,x) = h(x)$$
    Sol:
        $$\phi(t,x)=E^x[h(X_t)exp{int_0^t c(X_u) du}]$$
    with
        $$ dX_t = a(X_t) dt + b(X_t) db_t$$
    """
    a=StringFunction(a)
    b=StringFunction(b)
    c=StringFunction(c)
    h=StringFunction(h)

    Exnum = 100
    M = 100
    N = 50

    nx=M+1
    dx=(xn-x0)/float(M)
    xmesh = linspace(x0, xn, nx)

    nt=N+1
    dt=(tn-t0)/float(N)
    tmesh = linspace(t0, tn, nt)

    Ex=np.zeros([N+1,M+1])
    # Initialial Condition
    Ex[0,:]=h(xmesh)
    
    sqrtdt = sqrt(dt)

    for j in range(0,nx):
        bts = [0.0] * Exnum
        Xts = [xmesh[j]]   * Exnum
        Is  = [0.0] * Exnum
        for i in range(1,nt):
            Ex[i,j] = 0.0
            for k in range(0, Exnum):
                #bt = bts[k]
                Xt = Xts[k]
                Is[k] += c(Xt)* dt
                Ex[i,j] += h(Xt) * exp(Is[k])
                db = random.randn()*sqrtdt
                bts[k] += db
                Xts[k] += a(Xt)* dt + b(Xt) * db
            Ex[i,j] /= Exnum   
    return Ex

         

# ================================================================
# Feynman-Kac formula:
#
# Function:  phi(t,x).
#
#       dphi     f(x)^2 d^2 phi          dphi
# PDE:  ----  =  ------ -------  +  e(x) ----  +  g(x) phi
#        dt        2     dx^2             dx
#
# with initial conditions
#   phi(0,x) = h(x).
#
# Solution
#
#   phi(t,x) = E^x[ h(X_t) exp{int_0^t g(X_u) du} ]
#
# with
#
#   dX_t = e(X_t) dt + f(X_t) db_t
#
# where E^x[...] means E[... | X_0 = x].

def fksim_v(a,b,c,h,x0=-2.,xn=2.,t0=0.,tn=1.):
    """
    Feynman-Kac formula:
    PDE:
        $$ \frac{d\phi}{d t} = a(x) \frac{d\phi}{d x}+\frac{b^2}{2}\frac{d^2\phi}{d x^2}+c\phi$$
    IC:
        $$\phi(0,x) = h(x)$$
    Sol:
        $$\phi(t,x)=E^x[h(X_t)exp{int_0^t c(X_u) du}]$$
    with
        $$ dX_t = a(X_t) dt + b(X_t) db_t$$
    """
    Exnum = 100
    M=100
    N=10
    a=StringFunction(a)
    b=StringFunction(b)
    c=StringFunction(c)
    h=StringFunction(h)

    nx=M+1
    dx=(xn-x0)/float(M)
    

    nt=N+1
    dt=(tn-t0)/float(N)
    sqrtdt = sqrt(dt)
    
    xmesh = linspace(x0, xn, nx)
    tmesh = linspace(t0, tn, nt)

    Ex=np.zeros([N+1,M+1])
    # Initialial Condition
    Ex[0,:]=h(xmesh)
    """
    sample 1000
    vectorization calculation: 0.000061035625
    loop calculation: 0.00108408927917

    t0=time();
    ICset(Ex,h)
    tt=time()-t0
    print tt 
    """
    e = np.ones(Exnum,float)
    for j in range(0,nx):
        bts = np.zeros(Exnum,float)
        Xts = xmesh[j] * np.ones(Exnum,float)
        Is  = np.zeros(Exnum,float)
        Xt=Xts
        for i in range(1,nt):
            # vectorization calculation
            Ex_u = np.zeros(Exnum,float) 
            db = random.randn(Exnum)*sqrtdt
            Xts += e * a(Xt) * dt + e * b(Xt) * db
            Xt=Xts
        
            Is = e*c(Xt)*dt

            Ex_u[:] += e*h(Xt[:])*exp(Is[:])
            Ex[i,j] = np.sum(Ex_u)/Exnum
    return Ex
    #plot(xmesh,Ex[0,:],xmesh,Ex[N,:],dpi='1200',hardcopy='test.eps')
    """
    # making Animation
    counter = 0 
    for s in range(0,M):
        y = Ex[s,:]
        plot(xmesh,Ex[0,:],xmesh, y, axis=[xmesh[0], xmesh[-1],0.01,1.],
            xlabel='x', ylabel='fk', legend='s=%04d' % s,
            hardcopy='tmp_%04d.ps' % counter)
        counter += 1
    movie('tmp_*.ps' ,encoder='convert',fps=2, output_file='a.gif')
    """

def sdeplot(a,b,x0=1,t0=0,tn=1, n=200):
    T=tn-t0
    nt=n
    dt = T/nt
    sqrtdt = sqrt(dt)
    t = linspace(t0, tn, nt+1)    # 101 points between 0 and 3
    x_init=x0
    t_init=t0
    t_end=tn
    x=euler(a,b,x_init,t_init,t_end,nt)
    
    title1= 'dx=(%s)dt+(%s )dw' %(a,b)
    
    plot(t, x, 'r-6', \
         legend=(title1), \
         dpi=1200, hardcopy="test.png")
    
def sdeplot1(coeff,t0=0,x0=1,tn=1):
    [A,B,C,D]=coeff
    T=tn-t0
    nt=200
    dt = T/nt

    sqrtdt = sqrt(dt)
    #t=array([])
    t = linspace(t0, tn, nt+1)    # 101 points between 0 and 3
    e=ones([1,size(t)])
    
    x=0.*copy(t)
    y=0.*copy(t)
    dw=0.*copy(t)
    
    """
    for j in range(1, nt+1):
        dw  = random.gauss(0, sqrtdt)
        #w[j] = w[j-1] + dw
        x[j] = x[j-1]+(A+B*x[j-1])*dt \
        +(C+D*x[j-1])*dw 
    """
    """
    dw = random.randn(nt+1)*sqrtdt
    dw[0]=x0
    x=cumsum(dw)
    """
    x[0]=x0
    y[0]=x0
    for j in range(0, nt):
        #dw.append(random.gauss(0, sqrtdt))
        #dw[j]= gauss(0, sqrtdt)
        dw[j]=random.randn()*sqrtdt
        y[j+1]= y[j]+(A+B*y[j])*dt
        x[j+1]= x[j]+(A+B*x[j])*dt+(C+D*x[j])*dw[j]
    
    #x[1:nt+1] = x[0:nt]+(A+B*x[0:nt])*dt+(C+D*x[0:nt])*dw[0:nt]

    #            +[m*n for m,n in zip(C*e[0:nt]+D*x[0:nt],dw[0:nt])]
    
    """
    The classic part of SDE:
    dX_t= ( A + B X_t) dt
    """
    """
    if B==0:
       func=x0+A*t
    else: 
       func=-A/B+(A/B+x0)*exp(B*t) 
    """
    title1= 'dx=(%s+%s x)dt+(%s+%s x)dw' %(A,B,C,D)
    title2= 'dx=(%s+%s x)dt' %(A,B)
    plot(t, x, t, y, 'r-6', \
         legend=(title1,title2), \
         dpi=1200, hardcopy="test.png")
    
