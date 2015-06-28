# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:02:40 2015

@author: Edward
"""

"""Set up the Plot package"""

import os, glob, shutil
import matplotlib

__all__ = ['FigureData', 'PublicationFigures']

# First check under ./fonts/ttf that Helvectica family fonts exist
resourcepath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resource/Helvetica.ttf'))
rcpath = matplotlib.matplotlib_fname() # get rc file path
fontpath = os.path.abspath(os.path.join(os.path.dirname(rcpath), 'fonts/ttf/'))
targetpath = glob.glob(os.path.abspath(os.path.join(fontpath, 'Helvetica*.ttf')))
try: # try to copy the font to the ttf folder, if have write permission
    if not targetpath: # if font not found, copy from resource folder
        targetpath = os.path.abspath(os.path.join(fontpath, 'Helvetica*.ttf'))
        shutil.copy(resourcepath, targetpath) # may fail
        print('Helvetica.ttf is copied to %s' %targetpath)
        # Set up matplotlib properties
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
        matplotlib.rcParams['font.size'] = 12.0
except: # use alternative ways to set up the font
    # use a different backend, as the default backend may not work
    print('Do not have permission to copy the file. Use alternative solution')
    matplotlib.use('Agg') # use 'Agg' backend
    prop = matplotlib.font_manager.FontProperties(fname=resourcepath)    







