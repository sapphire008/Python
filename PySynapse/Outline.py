# -*- coding: utf-8 -*-
"""
Electrophysiology Data Acquisition GUI Development Outline

Created on Thu Feb 19 20:47:40 2015

@author: Edward


Inspired by MultiChannel System (R)

PART I: Hardware interfaces
 1). Connections
     i).    USB
     ii).   Serial Port
     iii).  VGA / DVI / HDMi
 2). Rack (classes)
     i).    Amplifier
     ii).   Pressure System
     iii).  Flow system
     iv).   Stimulation
     v).    Automatic glass electrode position calibration (new)
     vi).   Auto-microsocpe focus (new)
 3). Protocol: automatically run prespecified sequences of protocls 

PART II: Data visualization
 1). Oscilliscope
     i).     Seal Test
     ii).     Moving Window
     iii).    Digital Filters
     iv).     
 2). Camera
     i).    Image / Video I/O
     ii).   Image processing
         a). Affine transformations, zoom, pan
         b). Filter
         c). Denoise / Sharpen
         d). Contrast adjustment
         e). Marking / pen on image
         f). Feature recognition / extraction tools           

 PART III: Data preprocessing
 1). Data I/O: explore NEO
 2). Feature detection
     i).    EPSP / IPSP / EPSC / IPSC
     ii).   Action potential characterization
     iii).  After hyperpolarization / depolarization characterization
"""