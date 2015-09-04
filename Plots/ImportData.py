# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:24:54 2015

@author: Edward
"""

import numpy as np
import pandas as pd
import os
import zipfile

dataFile = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/example/lineplot.csv'

class Protocol(object): # for composition
    pass

class NeuroData(object):
    """Read electrophysiology data file
    """
    def __init__(self, dataFile=None, old=False, *args, **kwargs):
        """Initialize class"""
        self.Voltage = {}
        self.Current = {}
        self.Stimulus = {}
        self.Protocol = Protocol() # composition
        if dataFile is not None and isinstance(dataFile, str):
            # load directly if all the conditions are met
            self.LoadData(dataFile=dataFile, old=old, *args, **kwargs)
        else:
            IOError('Unrecognized data file input')

    def LoadData(self, dataFile, old=True, *args, **kwargs): #old=True to be edited later
        """Load data in text file"""
        dataFile = dataFile.replace('\\','/')# make sure using forward slash
        # check file exists
        if not os.path.isfile(dataFile):
            IOError('%s does not exist' %dataFile)
        # Evoke proper load method
        if old:
            self.LoadOldDataFile(dataFile, *args, **kwargs)
        else:
            self.LoadDataFile(dataFile, *args, **kwargs)

    def LoadDataFile(self, dataFile):
        """Read zipped data file (new format)"""
        archive = zipfile.ZipFile(dataFile, 'r')
        # Check if the file is a valid zipfile
        if not archive.is_zipfile():
            IOError('%s is not a valid zip file'%dataFile)
        # read header txt file
        fid = archive.read('header.txt','r')
        self.Protocol.infoBytes = np.fromfile(fid, np.int32, 1) # size of header
        # ... etc

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
                           'StimulusAmpA':'StimA',
                           'StimulusAmpB':'StimB',
                           'StimulusAmpC':'StimC',
                           'StimulusAmpD':'StimD',
                           'StimulusAmpA9':'StimA'}
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


""" 2photon image data"""
class ImageData(object):
    """Read image data file
    """
    def __init__(self, dataFile=None):
        """Initialize class"""
        self.Img = {}
        self.Protocol = {}
        if dataFile is not None and isinstance(dataFile, str):
            self.LoadImageData(dataFile)

""" Publication figure data (csv file or NeuroData file) """
class FigureData(object):
    def __init__(self, dataFile=None, *args, **kwargs):
        """Initialize class"""
        self.meta = {} # a list of meta parameters in the file
        if dataFile is None or not isinstance(dataFile, (str,list,tuple,np.ndarray)):
            return
        # load the file
        self.__loadbyext(dataFile, *args, **kwargs)
        
    def __loadbyext(self,  dataFile, ext=None, *args, **kwargs):
        """Load data based on extension"""
        if ext is None:
            f = dataFile[0] if isinstance(dataFile, (list,tuple,np.ndarray)) \
                                else dataFile
            ext = os.path.splitext(os.path.basename(os.path.abspath(f)))[-1]
        if ext == '.csv': # load csv text file which contains attributes of plots
            self.LoadFigureData(dataFile=dataFile, *args, **kwargs)
        elif ext == '.dat':
            # load NeuroData
            self.LoadNeuroData(dataFile, *args, **kwargs)
        else:
            raise(TypeError('Unrecognized extension %s'%(ext)))
        
    def LoadFigureData(self, dataFile, sep=',', metachar="|"):
        """Load data file"""
        if not isinstance(dataFile, str):
            raise(TypeError('Please give a single path to .csv data file'))
        fid = open(dataFile, 'r')
        self.table = []
        for rownum, line in enumerate(fid): # iterate through each line
            line = line.strip().strip(sep).replace('\t','').replace('"','')
            if not line or line[0] == "#" or line==line[0]*len(line):
                continue # skip comments and empty lines
            if line[0] == metachar: # metadata starts with "|"
                self.parse_meta(line, metachar)
            else: # assuming the rest of the file is data table
                fid.close()
                break
        # read in the rest of the data
        self.table = pd.read_csv(dataFile, sep=sep, comment="#",
                                 skipinitialspace=True, skiprows=rownum,
                                 skip_blank_lines=True)
        # set more parameters
        self.set_default_labels()
        
    def set_default_labels(self,cat=None):
        def copyvalue(meta, f, g, cat=None):
            if f not in meta.keys() and g in meta.keys():
                if isinstance(meta[g],list):
                    meta[f] = cat.join(meta[g]) if cat is not None else ""
                else:
                    meta[f] = meta[g]
            return(meta)
        self.meta = copyvalue(self.meta, 'xlabel','x', cat=cat)
        self.meta = copyvalue(self.meta, 'ylabel','y', cat=cat)
        try:
            self.meta = copyvalue(self.meta, 'zlabel','z', cat=cat)
        except:
            pass

    def parse_errorbar(self, df=None, simplify=True):
        """Reorganize errorbar"""
        if df is None:
            df = self.table
        # find columns of errorbar data
        keys = list(self.meta.keys())
        # Function to get errobar
        def PE(p):
            out = df[list(p)]
            if out.ndim == 1: # including cases where 'error' column is specified
                return([np.array(out), np.array(out)])
            elif out.ndim == 2 and out.shape[-1] == 2:
                return(np.array(out).T)
            else: # fall thorugh, should not happen
                return(None)
        if 'error_pos' in keys and 'error_neg' in keys:
            P = np.array([self.meta['error_pos'], self.meta['error_neg']]).T
        elif 'error_pos' in keys:
            P = np.array([self.meta['error_pos']])
        elif 'error_neg' in keys:
            P = np.array([self.meta['error_neg']])
        P = [P] if P.ndim==1 else P
        out = [PE(p) for p in P]
        out = out[0] if len(out)==1 and simplify else out
        return(out)
        
    def parse_meta(self, line,metachar="|"):
        """Parse parameter"""
        line = line.replace(metachar,"")
        m, v = line.split("=") # metavaraible, value
        m, v = m.strip(), v.strip()
        # parse value if it is a list
        if v.lower() == "none":
            self.meta[m] = None
            return
        if "[" in v and "," in v:
            v = v.replace("[","").replace("]","").split(",")
            v = [x.strip() for x in v]
        self.meta[m] = v
        # Force some default values
        if 'xlabel' not in self.meta.keys(): self.meta['xlabel'] = ''
        if 'ylabel' not in self.meta.keys(): self.meta['ylabel'] = ''
    
    def LoadNeuroData(self, dataFile, *args, **kwargs):
        self.Neuro2Trace(dataFile, *args, **kwargs)

    def Neuro2Trace(self, data, channels=None, streams=None, *args, **kwargs):
        """Use NeuroData method to load and parse trace data to be plotted
        data: an instance of NeuroData, ro a list of instances
        channels: list of channels to plot, e.g. ['A','C','D']
        streams: list of data streams, e.g. ['V','C','S']
        """
        # Check instance
        if isinstance(data, NeuroData):
            data = [data] # convert to list
        elif isinstance(data, str): # file path
            data = [NeuroData(data, *args, **kwargs)]
        elif isinstance(data, list): # a list of objects
            for n, d in enumerate(data): # transverse through the list
                if isinstance(d, NeuroData): # a list of NeuroData instances
                    pass
                elif isinstance(d,str): # a list of file paths
                    data[n] = NeuroData(d, *args, **kwargs)
                else:
                    raise TypeError(("Unrecognized data type"))
        else:
            raise TypeError(("Unrecognized data type"))

        # initialize notes, stored in stats attribute
        self.meta.update({'notes':[], 'xunit':[],'yunit':[],'x':[], 'y':[]})
        # file, voltage, current, channel, time
        notes = "%s %.1f mV %d pA channel %s WCTime %s min"
        self.table = []
        
        for n, d in enumerate(data): # iterate over all data
            series = pd.DataFrame() # initialize series data frame
            # Time data
            series['time'] = pd.Series(np.arange(0, d.Protocol.sweepWindow+
                          d.Protocol.msPerPoint, d.Protocol.msPerPoint))
            self.meta['x'].append('time')
            self.meta['xunit'].append('ms') # label unit
            # iterate over all the channels
            avail_channels = [x[-1] for x in d.Protocol.channelNames]
            avail_streams = [x[:-1] for x in d.Protocol.channelNames]
            for c in self.listintersect(channels, avail_channels):
                # iterate over data streams
                for s in self.listintersect(avail_streams,streams):
                    tmp = {'Volt': d.Voltage, 'Cur':d.Current, 'Stim': d.Stimulus}.get(s)
                    if tmp is None or not bool(tmp):
                        continue
                    tmp = tmp[c]
                    if tmp is None:
                        continue
                    series[s+c] = tmp # series[s, c]
                    self.meta['y'].append(s+c) # .append((s,c))
                    if s[0] == 'V':
                        self.meta['yunit'].append('mV')
                    elif s[0] == 'C':
                        self.meta['yunit'].append('pA')
                    else: #s[0] == 'S'
                        self.meta['yunit'].append('pA')
                dtime = self.sec2hhmmss(d.Protocol.WCtime)
                # Notes: file, voltage, current, channel, time
                notesstr = notes %(d.Protocol.readDataFrom,  \
                                d.Voltage[c][0], d.Current[c][0], c, dtime)
                self.meta['notes'].append(notesstr)
            self.table.append(series)
        
        # if only 1 data set in the input, output as a dataframe instead of a
        # list of dataframes
        self.table = self.table[0] if len(self.table)<2 else self.table
        # reshape y meta data
        #if len(self.meta['x'])!=len(self.meta['y']) and len(self.meta['x'])>1:
        self.meta['y']=np.reshape(self.meta['y'],(len(self.meta['x']),-1))
        self.meta['yunit'] = np.reshape(self.meta['yunit'],\
                                           (len(self.meta['xunit']),-1))

    @staticmethod
    def listintersect(*args):
        """Find common elements in lists"""
        args = [x for x in args if x is not None] # get rid of None
        def LINT(A,B):  #short for list intersection
            return list(set(A) & set(B))
        if len(args) == 0:
            return(None)
        elif len(args) == 1:
            return(args[0])
        elif len(args) == 2:
            return(LINT(args[0],args[1]))
        else:
            newargs = tuple([LINT(args[0], args[1])]) + args[2:]
            return(listintersect(*newargs))

    @staticmethod
    def sec2hhmmss(sec):
        """Converting seconds into hh:mm:ss"""
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return("%d:%d:%0.1f" % (h, m, s))


if __name__ == '__main__':
#    data = NeuroData(dataFile, old=True)
#    figdata = FigureData()
#    figdata.Neuro2Trace(data, channels=['A','B','C','D'], streams=['Volt','Cur','Stim'])
    data = FigureData(dataFile='D:/Data/2015/07.July/Data 2 Jul 2015/Neocortex C.02Jul15.S1.E40.dat',old=True, channels=['A'], streams=['Volt','Cur','Stim'])
