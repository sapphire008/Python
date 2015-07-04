# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:24:54 2015

@author: Edward
"""

import numpy as np
import os

dataFile = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/example/Cell B.10Feb15.S1.E28.dat'

class Protocol(object): # for composition
    pass

class NeuroData(object):
    """Read electrophysiology data file
    """
    def __init__(self, dataFile=None, old=False):
        """Initialize class"""
        self.Voltage = {}
        self.Current = {}
        self.Stimulus = {}
        self.Protocol = Protocol() # composition
        if dataFile is not None and isinstance(dataFile, str):
            # load directly if all the conditions are met
            self.LoadData(dataFile=dataFile, old=old) 
        else:
            IOError('Unrecognized data file input')

    def LoadData(self, dataFile, old=False):
        """Load data in text file"""
        # check file exists
        if not os.path.isfile(dataFile):
            IOError('%s does not exist' %dataFile)
        # Evoke proper load method
        if old:
            self.LoadOldDataFile(dataFile)
        else:
            self.LoadDataFile(dataFile)
            
    
    def LoadDataFile(self, dataFile):
        """Read zipped data file (new format)"""
        return
    
    def LoadOldDataFile(self, dataFile, numChannels=4, infoOnly=False):
        """Read Old .dat format data file"""
        
        self.Protocol.numChannels = numChannels # hard set
        self.Protocol.readDataFrom = os.path.abspath(dataFile).replace('\\','/') # store read location
        with open(dataFile, 'rb') as fid:
            fid.seek(6, 0) # set to position 6 from the beginning of the file
            self.Protocol.infoBytes = np.fromfile(fid, np.int32, 1) # size of header
            self.Protocol.sweepWindow = np.fromfile(fid, np.float32, 1)[0] #in msec per episode
            self.Protocol.msPerPoint = np.fromfile(fid, np.float32, 1)[0] / 1000.0 # in microseconds per channel, divided by 1000 to msec
            self.Protocol.numPoints = np.fromfile(fid, np.float32, 1)[0] # number of data points
            self.Protocol.WCtime = np.fromfile(fid, np.float32, 1)[0] # in seconds since went whole cell
            self.Protocol.drugTime = np.fromfile(fid, np.float32,1)[0] # in seconds since most recent drug started
            self.Protocol.drug = np.fromfile(fid,np.float32,1)[0] #an integer indicating what drug is on

            #% new from BWS on 12/21/08
            np.fromfile(fid,np.int32,1) # simulated data
            fid.seek(48 , 0)
            self.Protocol.genData = np.fromfile(fid, np.float32, 56) # [need expansion]

            # read in TTL information
            self.Protocol.ttlData = []
            for index in xrange(self.Protocol.numChannels):
                fid.seek(10, 1) # 10 is for VB user-defined type stuff
                self.Protocol.ttlData.append(np.fromfile(fid, np.float32, 17)) #[need expansion]
            #print(fid.tell())

            # read in DAC information
            self.Protocol.dacData = []
            self.Protocol.dacName = []
            for index in xrange(self.Protocol.numChannels):
                fid.seek(10, 1) # 10 is for VB user-defined type stuff
                self.Protocol.dacData.append(np.fromfile(fid, np.float32, 42)) #[need exspansion]
                self.Protocol.dacName.append(self.readVBString(fid))
            
            #print(fid.tell())
            # Get other parameters
            self.Protocol.classVersionNum = np.fromfile(fid, np.float32, 1)[0]
            self.Protocol.acquireComment=self.readVBString(fid)
            self.Protocol.acquireAnalysisComment=self.readVBString(fid)
            self.Protocol.drugName=self.readVBString(fid)
            self.Protocol.exptDesc=self.readVBString(fid)
            self.Protocol.computerName=self.readVBString(fid)
            self.Protocol.savedFileName=os.path.abspath(self.readVBString(fid)).replace('\\','/')
            self.Protocol.fileName = self.Protocol.savedFileName
            self.Protocol.linkedFileName=os.path.abspath(self.readVBString(fid)).replace('\\','/')
            self.Protocol.acquisitionDeviceName=self.readVBString(fid)
            self.Protocol.traceKeys=self.readVBString(fid)
            self.Protocol.traceInitValuesStr=self.readVBString(fid)
            self.Protocol.extraScalarKeys=self.readVBString(fid)
            self.Protocol.extraVectorKeys=self.readVBString(fid)
            self.Protocol.genString=self.readVBString(fid)
            self.Protocol.TTLstring = []
            for index in xrange(self.Protocol.numChannels):
                self.Protocol.TTLstring.append(self.readVBString(fid))
            self.Protocol.ampDesc = []
            for index in xrange(self.Protocol.numChannels):
                self.Protocol.ampDesc.append(self.readVBString(fid))
                
            # Get Channel info
            channelDict = {'VoltADC1':'VoltA','VoltADC3':'VoltB',
                           'VoltADC5':'VoltC','VoltADC7':'VoltD',
                           'CurADC0':'CurA','CurADC2':'CurB',
                           'CurADC4':'CurC','CurADC6':'CurD',
                           'StimulusAmpA':'StimulusA',
                           'StimulusAmpB':'StimulusB',
                           'StimulusAmpC':'StimulusC',
                           'StimulusAmpD':'StimulusD',
                           'StimulusAmpA9':'StimulusA'}
            keys = [k.split("/")[0] for k in self.Protocol.traceKeys.split()]
            self.Protocol.channelNames = [channelDict[k] for k in keys]
            self.Protocol.numTraces = len(self.Protocol.channelNames)
            
            if infoOnly: # stop here if only 
                return            
            
            # Read trace data
            self.Protocol.traceDesc = []   
            for chan in self.Protocol.channelNames:
                traceFactor = float(np.fromfile(fid, np.float32, 1))
                traceLength = int(np.fromfile(fid, np.int32, 1))
                traceDesc = self.readVBString(fid)
                self.Protocol.traceDesc.append(traceDesc)
                traceData = np.fromfile(fid, np.int16, traceLength)
                traceData = traceFactor * traceData
                if chan[0] == 'V':
                    self.Voltage[chan[-1]] = traceData
                elif chan[0] == 'C':
                    self.Current[chan[-1]] = traceData
                elif chan[0] == 'S':
                    self.Stimulus[chan[-1]] = traceData
                else: # fallthrough
                    TypeError('Unrecognized channel type')
        
        # close file
        fid.close()
        
    @staticmethod
    def readVBString(fid):
        stringLength = int(np.fromfile(fid, np.int16, 1))
        if stringLength==0:
            return('')
        else:
            return(''.join(np.fromfile(fid, '|S1', stringLength)))
            
            

class ImageData(object):
    """Read image data file
    """
    def __init__(self, dataFile=None):
        """Initialize class"""
        self.Img = {}
        self.Protocol = {}
        if dataFile is not None and isinstance(dataFile, str):
            self.LoadImageData(dataFile)

class FigureData(object):
    """Data for plotting
    """
    def __init__(self, dataFile=None):
        """Initialize class"""
        self.series = {'x':[],'y':[],'z':[]} 
        self.stats = {'x':{},'y':{},'z':{}}
        self.names = {'x':[],'y':[],'z':[]}
        self.num = {'x':[],'y':[],'z':[]} # count number of data sets
        if dataFile is not None and isinstance(dataFile, str):
            self.LoadFigureData(dataFile)        
            
    def LoadFigureData(self, dataFile, delimiter=','):
        """Read text data for figure plotting"""
          # check file exists
        if not os.path.isfile(dataFile):
            IOError('%s does not exist' %dataFile)
            
        with open(dataFile, 'rb') as fid:
            for line in fid: # iterate each line
                if not line.strip() or line[0] == "#":
                    continue  # skip comments
                # split comma delimited string
                # series code, series name,@datatype, data1, data2, data3, ...
                lst = [s.strip() for s in line.split(delimiter)]
                # Parse variable
                v = lst[0][0] # variable name
                stats = lst[0][1:-1]
                # Read the data
                seriesData = self.ParseFigureData(lst[1][1:], lst[3:])
                # Organize the data to structure
                if stats != "": #stats, not empty
                    if stats in self.stats[v].keys(): # key exists already
                        self.stats[v][stats].append(seriesData)
                    else: # add new key / create new list
                        self.stats[v][stats] = [seriesData]
                else: # series data
                    self.series[v].append(seriesData)
                    self.names[v].append(lst[2][1:-1])

            fid.close()
            # Parse number of data set
            for v in self.series.keys():
                self.num[v] = len(self.series[v])

    @staticmethod
    def ParseFigureData(valueType, seriesList):
        """Parse each line of read text"""
        if valueType == 'str':
            return(np.array(seriesList))
        elif valueType == 'float':
            return(np.array(seriesList).astype(np.float))
        elif valueType == 'int':
            return(np.array(seriesList).astype(np.int))
        else: # unrecognized type
            TypeError('Unrecognized data type')
            
    def Neuro2Figure(self, data, channels=None, streams=None):
        """Use NeuroData method to load and parse trace data to be plotted
        data: an instance of NeuroData, ro a list of instances
        channels: list of channels to plot, e.g. ['A','C','D']
        streams: list of data streams, e.g. ['V','C','S']
        """
        # Check instance
        if isinstance(data, NeuroData):
            data = [data] # convert to list

        # initialize notes, stored in stats attribute        
        self.stats['y']['notes'] = []
        notes = "%s %.1f mV %d pA channel %s WCTime %s min"
        for n, d in enumerate(data): # iterate over all data
             # Time data
            self.series['x'].append(np.arange(0, d.Protocol.sweepWindow+
                          d.Protocol.msPerPoint, d.Protocol.msPerPoint))
            self.names['x'].append('ms') # label unit
             # iterate over all the channels
            for c in channels:
                # iterate over data streams
                for s in streams:
                    tmp = {'V': d.Voltage, 'C':d.Current, 'S': d.Stimulus}.get(s)
                    if tmp is None or not bool(tmp):
                        continue
                    tmp = tmp[c]
                    if tmp is None:
                        continue
                    self.series['y'].append(tmp)
                    self.names['y'].append((s, c))
                    if s == 'V':
                        volt_i = tmp[0]
                    if s == 'C':
                        cur_i = tmp[0]
                dtime = self.sec2hhmmss(d.Protocol.WCtime)
                # Notes: {path} Initial: {voltage 0.1f mV} {current %d pA} \
                #                   WC Time: 1:12:30.0 min 
                notesstr = notes %(d.Protocol.readDataFrom,  volt_i, cur_i, c,dtime)
                self.stats['y']['notes'].append(notesstr)
    
    @staticmethod
    def sec2hhmmss(sec):
         m, s = divmod(sec, 60)
         h, m = divmod(m, 60)
         return("%d:%d:%0.1f" % (h, m, s))

      
            
if __name__ == '__main__':
    data = NeuroData(dataFile, old=True)
    figdata = FigureData()
    figdata.Neuro2Figure(data, channels=['A','C','D'], streams=['V','C','S'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    