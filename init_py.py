# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:44:12 2016

@author: Edward
"""

# matplotlib.use('PS')
import sys
import os

# parsing mode
if len(sys.argv)>1:
    if any(['ipy' in argv.lower() for argv in sys.argv]) or \
            any(['jupyter' in argv.lower() for argv in sys.argv]) or \
            __file__ in sys.argv:  # %run magic command
        MODE = 'jupyter'
    else: # assume this is called from the command line
        MODE = sys.argv[1]
else:
    MODE = 'academic'

import matplotlib
if sys.platform == 'darwin':
    PYPKGPATH = os.path.dirname(os.path.realpath(__file__))
else:
    PYPKGPATH = 'D:/Edward/Documents/Assignments/Scripts/Python/'
# PYPKGPATH = '/Users/edward/Documents/Scripts/Python'
sys.path.append(PYPKGPATH)
import glob
pkg_list = [os.path.basename(os.path.normpath(p)) for p in glob.glob(PYPKGPATH+'/*') if os.path.isdir(p)]

def addpythonpkg(pkg=None, recursively=False):
    """Instead of doing complicated import, simply add the path of custom scripts"""
    if pkg is None:
        # print(pkg_list)
        return pkg_list
    if isinstance(pkg, str):
        pkg = [pkg]
    # add packages
    for p in pkg:
        if p in pkg_list and p not in sys.path:
            if recursively:
                [sys.path.append(os.path.join(PYPKGPATH, x[0])) for x in os.walk(p)]
            else:
                sys.path.append(os.path.join(PYPKGPATH, p))
        else:
            raise(Exception('package %s is not available. Check spelling'%p))


def getaddedpkgs():
    """List packages that has been added"""
    added_paths = []
    for p in sys.path:
        normpath = os.path.basename(os.path.normpath(p))
        if normpath in pkg_list and normpath not in added_paths:
            added_paths.append(normpath)

    return added_paths

def getaddedmodules():
    return sys.modules.keys()

#def cd(path=None):
#    """overloading cd"""
#    if not isinstance(path,str):
#        return
#    elif path.lower()== 'home':
#        os.chdir(PYPKGPATH)
#    elif path.lower()== '..':
#        os.chdir(os.path.dirname(os.getcwd()))
#    else:
#        os.chdir(path)

global tableau10
global FONT
global FONTSIZE

if MODE == 'academic':
    print('academic mode')
    # Do some generic import
    addpythonpkg(['Spikes', 'generic', 'Plots'])
    import pandas as pd
    import numpy as np
    from spk_util import *
    from MATLAB import *
    from ImportData import *
    from plots import *
    # Added the platelet tableau10
    tableau10 = ["#1f77b4","#ff7f0e", "#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    FONT = 'D:/Edward/Documents/Assignments/Scripts/Python/PySynapse/resources/Helvetica.ttf'
    FONTSIZE =  {'title':10, 'xlab':8, 'ylab':8, 'xtick':6,'ytick':6,'texts':6,
            'legend': 6, 'legendtitle':6, 'xminortick':6, 'yminortick':6} # font size
elif MODE == 'jupyter':
    print('jupyter mode')
    addpythonpkg(['generic', 'Plots'])
    import pandas as pd
    import numpy as np
    from MATLAB import *
    from plots import *
    tableau10 = ["#1f77b4","#ff7f0e", "#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    FONT = 'D:/Edward/Documents/Assignments/Scripts/Python/PySynapse/resources/Helvetica.ttf'
    FONTSIZE =  {'title':12, 'xlab':10, 'ylab':10, 'xtick':8,'ytick':8,'texts':8,
            'legend': 8, 'legendtitle':8, 'xminortick':8, 'yminortick':8} # font size
