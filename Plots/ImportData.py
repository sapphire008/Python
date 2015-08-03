# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:24:54 2015

@author: Edward
"""

import numpy as np
import pandas as pd
import os
import zipfile

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

    def LoadData(self, dataFile, old=True): #old=True to be edited later
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

""" Publication figure data (csv file) """
class FigureData(object):
    def __init__(self, dataFile=None):
        """Initialize class"""
        self.series = {'x':[],'y':[],'z':[]}
        self.meta = {} # a list of meta parameters in the file

        if dataFile is not None and isinstance(dataFile, str):
            # load directly if all the conditions are met
            self.LoadFigureData(dataFile=dataFile)

    def LoadFigureData(self, dataFile, sep=',', metachar="|"):
        """Load data file"""
        # check file exists
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
        # parse the data to variable
        self.parse_data()
        # set more parameters
        self.set_default_labels()
        # Reorganize errorbar
        self.parse_error_bar()
        # Count number of data sets for each series
        self.count_series()
        self.get_shape()
        self.get_size()

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
        if self.series['z']:
            self.meta = copyvalue(self.meta, 'zlabel','z', cat=cat)

    def parse_error_bar(self):
        """Reorganize errorbar"""
        vec = [0,0,0,0,0,0]
        vec[0] = 1 if 'errorbar_pos' in self.series.keys() else 0
        if vec[0] == 1:
            vec[1] = 1 if self.series['errorbar_pos'] else 0
        vec[2] = 1 if 'errorbar_neg' in self.series.keys() else 0
        if vec[2] == 1:
            vec[3] = 1 if self.series['errorbar_neg'] else 0
        vec[4] = 1 if 'errorbar' in self.series.keys() else 0
        if vec[4] == 1:
            vec[5] = 1 if self.series['errorbar'] else 0
        # both pos and neg exist, but not errorbar
        if [vec[v] for v in [0,1,2,3,5]] == [1,1,1,1,0]:
            self.series['errorbar'] = [np.array([x,y]) for x,y in zip(\
                   self.series['errorbar_pos'], self.series['errorbar_neg'])]
        # pos exist but not neg
        elif [vec[v] for v in [0,1,3,5]] == [1,1,0,0]:
            self.series['errorbar'] = self.series['errorbar_pos']
        # neg exist but not pos
        elif [vec[v] for v in [1,2,3,5]] == [0,1,1,0]:
            self.series['errorbar'] == self.series['errorbar_neg']

        # remove errorbar_pos and errorbar_neg
        self.series.pop('errorbar_pos',None)
        self.series.pop('errorbar_neg',None)

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

    def parse_data(self):
        """Put table data into series"""
        # series
        cheader = list(self.table.columns.values)
        for key,val in iter(self.meta.items()):
            #if key in ['x','y','z']: continue
            if val is None:
                continue
            elif isinstance(val, str):
                if val not in cheader:
                    continue
                else:
                    self.series[key].append(np.array(\
                                        list(self.table[val].values)))
            elif all([v in cheader for v in val]): # assume it is a list
                self.series[key] = []
                for v in val:
                    self.series[key].append(np.array(list(
                                                    self.table[v].values)))
            elif any([v in cheader for v in val]):
               raise SyntaxError(\
                     "Some header reference to '%s' is spelled wrong"% (key))
            #else:
                #raise RuntimeError(\
                #        "Fall through. Check (key, value)=(%s, %s)"%(key,val))
    def get_size(self):
        """Get the numel of each data series except the nan or empty string"""
        def checknumel(v):
            if isinstance(v[0], np.string_): # check for empty
                return(len([x for x in v if x]))
            else:
                try:
                    return(np.count_nonzero(~np.isnan(v)))
                except:
                    raise TypeError("unrecognized type %s" %(type(v)))

        self.size = {}
        for k in self.series.keys():
            self.size[k] = [checknumel(v) for v in self.series[k]]

    def get_shape(self):
        """Get shape of each series"""
        self.shape = {}
        for k in self.series.keys():
            self.shape[k] = [v.shape for v in self.series[k]]

    def count_series(self):
        """Get number of series"""
        self.num = {}
        for k in self.series.keys():
            self.num[k] = len(self.series[k])

    def Neuro2Trace(self, data, channels=None, streams=None):
        """Use NeuroData method to load and parse trace data to be plotted
        data: an instance of NeuroData, ro a list of instances
        channels: list of channels to plot, e.g. ['A','C','D']
        streams: list of data streams, e.g. ['V','C','S']
        """
        # Check instance
        if isinstance(data, NeuroData):
            data = [data] # convert to list
        elif isinstance(data, str): # file path
            data = [NeuroData(data)]
        elif isinstance(data, list): # a list of objects
            if isinstance(data[0], NeuroData): # a list of NeuroData instances
                pass
            elif isinstance(data[0],str): # a list of file paths
                data = [NeuroData(d) for d in data]
            else:
                raise TypeError(("Unrecognized data type"))
        else:
            raise TypeError(("Unrecognized data type"))

        # initialize notes, stored in stats attribute
        self.meta.update({'notes':[], 'xunit':[],'yunit':[],'x':[], 'y':[]})
        # file, voltage, current, channel, time
        notes = "%s %.1f mV %d pA channel %s WCTime %s min"
        for n, d in enumerate(data): # iterate over all data
             # Time data
            self.series['x'].append(np.arange(0, d.Protocol.sweepWindow+
                          d.Protocol.msPerPoint, d.Protocol.msPerPoint))
            self.meta['x'].append('time')
            self.meta['xunit'].append('ms') # label unit
            # iterate over all the channels
            avail_channels = [x[-1] for x in d.Protocol.channelNames]
            avail_streams = [x[0] for x in d.Protocol.channelNames]
            for c in self.listintersect(channels, avail_channels):
                # iterate over data streams
                for s in self.listintersect(streams, avail_streams):
                    tmp = {'V': d.Voltage, 'C':d.Current, 'S': d.Stimulus}.get(s)
                    if tmp is None or not bool(tmp):
                        continue
                    tmp = tmp[c]
                    if tmp is None:
                        continue
                    self.series['y'].append(tmp)
                    self.meta['y'].append((s, c))
                    if s == 'V':
                        volt_i = tmp[0]
                        self.meta['yunit'].append('mV')
                    elif s == 'C':
                        cur_i = tmp[0]
                        self.meta['yunit'].append('pA')
                    else: #s == 'S'
                        self.meta['yunit'].append('pA')
                dtime = self.sec2hhmmss(d.Protocol.WCtime)
                # Notes: file, voltage, current, channel, time
                notesstr = notes %(d.Protocol.readDataFrom,  volt_i, cur_i, c,dtime)
                self.meta['notes'].append(notesstr)

    @staticmethod
    def listintersect(*args):
        """Find common elements in lists"""
        args = [x for x in args if x is not None]
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
    data = NeuroData(dataFile, old=True)
    figdata = FigureData()
    figdata.Neuro2Trace(data, channels=['A','B','C','D'], streams=['V','C','S'])