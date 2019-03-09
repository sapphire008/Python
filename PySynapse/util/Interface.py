# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:55:04 2017

XMessage Interface for windows.
Allows to load variable from the current MATLAB session.

@author: Edward
"""

import os
import ctypes
import matlab.engine

class MATLABListener(object):
    def __init__(self):
        # ml = matlab.engine.start_matlab()
        future = matlab.engine.connect_matlab(async=True)
        eng = future.result()
        # eng.sqrt(4.0) # calls MATLAB's function

class XDMessage(object):
    lib_path = 'D:/Edward/Documents/Assignments/Scripts/Python/PySynapse/resources/lib/XDMessaging.dll'
    def __init__(self, lib_path=None):
        self.lib_path = lib_path
        if lib_path is not None:
            self.lib = ctypes.cdll.LoadLibrary(lib_path)
    
    def makeListener(self):
        listener = self.lib.XDListener()
        return
    
    def makeBroadcaster(self):
        client = self.lib.XDMessagingClient()
        return