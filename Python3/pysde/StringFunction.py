#!/usr/bin/env python
"""Make a string mathematical expression behave as a Python function."""

from math import *

class StringFunction1x:
    """
    Make a string expression behave as a Python function
    of one variable.
    Examples on usage:
    >>> from StringFunction import StringFunction1x
    >>> f = StringFunction1x('sin(3*x) + log(1+x)')
    >>> p = 2.0; v = f(p)  # evaluate function
    >>> p, v
    (2.0, 0.8191967904691839)
    >>> f = StringFunction1x('1+t', independent_variable='t')
    >>> v = f(1.2)  # evaluate function of t=1.2
    >>> print("%.2f" % v)
    2.20
    >>> f = StringFunction1x('sin(t)')
    >>> v = f(1.2)  # evaluate function of t=1.2
    Traceback (most recent call last):
        v = f(1.2)
    NameError: name 't' is not defined
    >>> f = StringFunction1x('a+b*x', set_parameters='a=1; b=4')
    >>> f(2)   # 1 + 4*2
    9
    """
    def __init__(self, expression,
                 independent_variable='x',
                 set_parameters=''):
        self._f = expression
        self._var = independent_variable  # 'x', 't' etc.
        self.__name__ = self._f  # class name = function expression
        self._code = set_parameters
        
    def set_parameters(self, code):
        self._code = code

    def __call__(self, x):
        # assign value to independent variable:
        exec('%s = %g' % (self._var, x))
        # execute some user code (defining parameters etc.):
        if self._code:  exec(self._code)
        return eval(self._f)  # evaluate function expression
    
class StringFunction_alt(StringFunction1x):
    """
    Extension of class StringFunction1 to an arbitrary
    number of independent variables.
    
    Example on usage:
    
    >>> from StringFunction import StringFunction_alt
    >>> f = StringFunction_alt('1+sin(2*x)')
    >>> f(1.2)
    1.675463180551151
    >>> f = StringFunction_alt('1+sin(2*t)', independent_variables='t')
    >>> f(1.2)
    1.675463180551151
    >>> f = StringFunction_alt('1+A*sin(w*t)', independent_variables='t', \
                               set_parameters='A=0.1; w=3.14159')
    >>> f(1.2)
    0.9412217323869594
    >>> f.set_parameters('A=1; w=1')
    >>> f(1.2)
    1.9320390859672263
    >>> # function of two variables:
    >>> f = StringFunction_alt('1+sin(2*x)*cos(y)', \
                               independent_variables=('x','y'))
    >>> f(1.2,-1.1)
    1.3063874788637866
    >>> f = StringFunction_alt('1+V*sin(w*x)*exp(-b*t)', \
                               independent_variables=('x','t'))
    >>> f.set_parameters('V=0.1; w=1; b=0.1')
    >>> f(1.0,0.1)
    1.0833098208613807
    """

    def __init__(self, expression,
                 independent_variables='x',
                 set_parameters=''):
        StringFunction1x.__init__(self, expression,
                                  independent_variables,
                                  set_parameters)

    def __call__(self, *args):
        # assign value to independent variable(s):
        vars = str(tuple(self._var)).replace("'", "") # remove quotes
        cmd = '%s = %s' % (vars, args)
        exec(cmd)
        # execute some user code (defining parameters etc.):
        if self._code:  exec(self._code)
        return eval(self._f)  # evaluate function expression

class StringFunction1:
    """
    Make a string expression behave as a Python function
    of one variable.
    Examples on usage:
    >>> from StringFunction import StringFunction1x
    >>> f = StringFunction1('sin(3*x) + log(1+x)')
    >>> p = 2.0; v = f(p)  # evaluate function
    >>> p, v
    (2.0, 0.8191967904691839)
    >>> f = StringFunction1('1+t', independent_variable='t')
    >>> v = f(1.2)  # evaluate function of t=1.2
    >>> print("%.2f" % v)
    2.20
    >>> f = StringFunction1('sin(t)')
    >>> v = f(1.2)  # evaluate function of t=1.2
    Traceback (most recent call last):
        v = f(1.2)
    NameError: name 't' is not defined
    >>> f = StringFunction1('a+b*x', a=1, b=4)
    >>> f(2)   # 1 + 4*2
    9
    >>> f.set_parameters(b=0)
    >>> f(2)   # 1 + 0*2
    1
    """
    def __init__(self, expression, **kwargs):
        self._f = expression
        self._var = kwargs.get('independent_variable', 'x') # 'x', 't' etc.
        self.__name__ = self._f  # class name = function expression
        self._prms = kwargs
        try:
            del self._prms['independent_variable']
        except:
            pass
        self._f_compiled = compile(self._f, '<string>', 'eval')
        
    def set_parameters(self, **kwargs):
        self._prms.update(kwargs)

    def __call__(self, x):
        # include indep. variable in dictionary of function parameters:
        self._prms[self._var] = x
        # evaluate function expression:
        #return eval(self._f, globals(), self._prms)
        return eval(self._f_compiled, globals(), self._prms)
    
class StringFunction(StringFunction1):
    """
    Extension of class StringFunction1 to an arbitrary
    number of independent variables.

    Example on usage:

    >>> from StringFunction import StringFunction
    >>> f = StringFunction('1+sin(2*x)')
    >>> f(1.2)
    1.675463180551151
    >>> f = StringFunction('1+sin(2*t)', independent_variables='t')
    >>> f(1.2)
    1.675463180551151
    >>> f = StringFunction('1+A*sin(w*t)', independent_variables='t', \
                           A=0.1, w=3.14159)
    >>> f(1.2)
    0.9412217323869594
    >>> f.set_parameters(A=1, w=1)
    >>> f(1.2)
    1.9320390859672263
    >>> # function of two variables:
    >>> f = StringFunction('1+sin(2*x)*cos(y)', \
                           independent_variables=('x','y'))
    >>> f(1.2,-1.1)
    1.3063874788637866
    >>> f = StringFunction('1+V*sin(w*x)*exp(-b*t)', \
                           independent_variables=('x','t'))
    >>> f.set_parameters(V=0.1, w=1, b=0.1)
    >>> f(1.0,0.1)
    1.0833098208613807
    >>> # vector field of x and y:
    >>> f = StringFunction('[a+b*x,y]', \
                           independent_variables=('x','y'))
    >>> f.set_parameters(a=1, b=2)
    >>> f(2,1)  # [1+2*2, 1]
    [5, 1]
    
    """
    def __init__(self, expression, **kwargs):
        StringFunction1.__init__(self, expression, **kwargs)
        self._var = tuple(kwargs.get('independent_variables', 'x'))
        try:    del self._prms['independent_variables']
        except: pass
        
    def __call__(self, *args):
        # add independent variables to self._prms:
        for name, value in zip(self._var, args):
            self._prms[name] = value
        try:
            return eval(self._f_compiled, globals(), self._prms)
        except NameError(msg):
            raise NameError('%s; set its value by calling set_parameters' % msg)

def _test():
    import doctest, StringFunction
    return doctest.testmod(StringFunction)

def _try():
    f = StringFunction1('a+b*sin(x)', independent_variable='x', a=1, b=4)
    print(f(2))
    f.set_parameters(a=-1, b=pi)
    print(f(1))

if __name__ == '__main__':
    _test()
