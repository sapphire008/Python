# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:13:39 2015

@author: Edward
"""

import time

# Super classes
class Rack(object):
    def __init__(self, name, params):
        self.name = name # name of the Rack
        self.params = params # basic paramters, usually as a dictionary
    def __str__(self): # for print
        return "%s is a(n) %s" % (self.name, self.params)

# Amplifiers for electrodes
class Amplifier(Rack):
    def __init__(self, ):
        Rack.__init__(self, "Amplifier", {})

# Pressure system for glass pipette
class Pressure(Rack):
    def __init__(self, params):
        Rack.__init__(self, "Pressure System", params)
        
# Flow system        
class Flow(Rack):
    def __init__(self, flow_rate = 1.5, temperature = 30):
        Rack.__init__(self,"Flow System", {})
        self.flow_rate = flow_rate # in mL / min
        self.temperature = temperature # in degree celcius
        self.state = 0 # 0 for safe, 1 high temperature, 2 for abnormal flow

# Stimulation, single or multi-channel electrodes
class Stimulation(Rack):
    def __init__(self, amp = 0, onset = 1000, duration = 20, interval = 500, 
                 times = 2, channel = 1):
        Rack.__init__(self,"Stimulation", {})
        self.amp = amp # voltage amplitude, in mV
        self.onset = onset # onset time, in ms, 
                           # with respect to the acquisition episode
        self.duration = duration # duration of the stimulatin, in microsec
        self.interval = interval # between-stimulation interval, in ms
        self.times = times # number of stimulation
        self.channel = channel # specify a number of multiple channels
        
# Real time monitoring of data stream, import other Python modules        
class Oscilliscope(Rack):
    def __init__(self, num_displays = 1, x_range=500, y_range= None):
        Rack.__init__(self, {})
        self.num_displays = num_displays
        self.x_range = x_range# horizontal range, in ms
        self.y_range = y_range# vertical range, in mV
        
# Acquisition Protocol
class Protocol(Rack):
    def __init__(self, recording_channels,  stimulation_channels, fs, acq_dur, 
                 acq_delay=-1, repeat=1, rep_interval = 5000, 
                 filter_class=None, drug = None, number = 1,
                 name = "Protocol-%s" % (time.strftime("%Y.%m.%d.%H:%M:%S"))):
        Rack.__init__(self, "Acquisition Protocol", {})
        self.recording_channels = recording_channels # channels of recording
        self.stimulation_channels = stimulation_channels # channels of stim
        self.fs = fs # sampling rate
        self.acq_dur = acq_dur # acquisition duration (ms)
        self.acq_delay = 0 # delay acquisition for ms
        self.repeat = repeat # number of times to repeat the protocol
        self.rep_interval = rep_interval # ms to wait between each repeat
        self.filter_class = filter_class # digital filter the data
        # drug used in acquisition, "C" = control; (drugname, dose, unit) tuple
        self.drug = ("C",0,"uM") if durg is None else drug 
        self.number = number # Ordinal number of the protocol
        self.name = name # name of the protocol
 

class Camera(Rack):
  
class GlassPipettePositioning(Rack):

class MicroscopeAutoFocus(Rack):


    

    