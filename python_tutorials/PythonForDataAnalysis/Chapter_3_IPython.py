# -*- coding: utf-8 -*-
"""
Created on Sun May 04 00:07:53 2014

@author: Edward
"""

# Poor man's debugger functions
def set_trace():
    # put this function right before the exception occurs to debug. Press "c"
    # to resume the program;
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)
    
def debug(f,*args, **kwargs):
    # allows arbitrarily calling debugger for a function. Press "c" to resume
    # the function; press "s" to step through each line of the function
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme='Linux')
    return pdb.runcall(f,*args, **kwargs)
    
# example usage of debug function
def f(x,y,z=1):
    tmp=x+y
    tmp= x*tmp*y
    return tmp/z

debug(f,1,2,z=3)


# Checking running time within IPython
strings = ['foo','foobar','baz','qux','python', 'Guido Van Rossum'] *100000
%time method1 = [x for x in strings if x.startswith('foo')]# get timing only once
%timeit method2 = [x for x in strings if x[:3]=='foo']# get the best time


# Profiling programs: %prun and %run -p
# example: 
import numpy as np
from numpy.linalg import eigvals
def run_experiment(niter=100):
    K = 100
    results = []
    for _ in xrange(niter):
        mat = np.random.randn(K, K)
        max_eigenvalue = np.abs(eigvals(mat).max())
        results.append(max_eigenvalue)
    return results
    
some_results = run_experiment()
print 'Largest one we saw %s' % np.max(some_results)
    
# run the above example with profiling: suppose the above scrips is in cprof_example.py
python -m cProfile -s cprof_example.py

# prun will profile a python statement rather than a .py file
%prun -l 7 -s cumulative run_experiment()

# reloading module dependencies: unlike MATLAB, Python does not reload changes
# automatically. To refresh the modified dependencies, use
import some_lib
reload(some_lib)
# IPython have deep/recursive reload if there are deeper dependency changes
dreload(some_lib)
    
# This concludes today's study