# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:18:36 2015

@author: Edward
"""
import os, sys
sys.path.append(os.path.dirname(__file__))
from PublicationFigures import PublicationFigures

 # example
plotType = 'barplot' # barplot, lineplot, trace, beeswarm
Style = 'Vstack' # for lineplot: Vstack, Twin

# Get example folder
exampleFolder = str(os.path.abspath(os.path.join(__file__,'../example/')))
dataFile = os.path.join(exampleFolder, '%s.txt' %plotType)

# Load data
K = PublicationFigures(dataFile=dataFile, SavePath=os.path.join(exampleFolder,'%s.png'%plotType))

# Selecct plot
if plotType == 'lineplot':
    # Line plot example
    K.LinePlot(Style=Style)
    K.axs[0].set_ylim([0.5,1.5])
    K.axs[1].set_ylim([0.05, 0.25])
elif plotType == 'beeswarm':
    # Beeswarm example
    K.Beeswarm()
    #K.AnnotateOnGroup(m=[0,1])
    #K.AnnotateBetweenGroups(0, 1, text='p=0.01234') # between 1st and 2nd group
elif plotType == 'trace':
    # Time series example
    K.Traces()
elif plotType == 'barplot':
    K.BarPlot()
    
# Final clean up
K.SetFont() # change to specified font properties
K.fig.set_size_inches(9, 6) # set it for now.
K.Save()