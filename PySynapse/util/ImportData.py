# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:24:54 2015

@author: Edward
"""

import numpy as np
import pandas as pd
import os
import zipfile
import six
import re
from pdb import set_trace
from collections import OrderedDict
# import matplotlib.pyplot as plt

def readVBString(fid):
    stringLength = int(np.fromfile(fid, np.int16, 1))
    if stringLength==0:
        return('')
    else:
        # if python 2
        if six.PY2:
            return(''.join(np.fromfile(fid, '|S1', stringLength)))
        # if python 3
        elif six.PY3:
            tmp = np.fromfile(fid, '|S1', stringLength)
            return(np.ndarray.tostring(tmp).decode('UTF-8'))

def isempty(m):
    """Return true if:
    a). an empty string
    b). a list of length zero
    c). a tuple of length zero
    d). a numpy array of length zero
    e). a singleton that is not None
    """
    if isinstance(m, (list, tuple, str, np.ndarray)):
        return len(m) == 0
    else:
        return True if m else False

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

    def LoadDataFile(self, dataFile, infoOnly=False, getTime=False):
        """Read zipped data file (new format)"""
        archive = zipfile.ZipFile(dataFile, 'r')
        # Check if the file is a valid zipfile
        if not archive.is_zipfile():
            IOError('%s is not a valid zip file'%dataFile)
        # read header txt file
        fid = archive.read('header.txt','r')
        self.Protocol.infoBytes = np.fromfile(fid, np.int32, 1) # size of header
        # ... etc

    def LoadOldDataFile(self, dataFile, numChannels=4, infoOnly=False, getTime=False):
        """Read Old .dat format data file"""
        self.Protocol.numChannels = numChannels # hard set
        self.Protocol.readDataFrom = os.path.abspath(dataFile).replace('\\','/') # store read location
        with open(dataFile, 'rb') as fid:
            fid.seek(6, 0) # set to position 6 from the beginning of the file
            self.Protocol.infoBytes = np.fromfile(fid, np.int32, 1) # size of header
            self.Protocol.sweepWindow = np.fromfile(fid, np.float32, 1)[0] #in msec per episode
            self.Protocol.msPerPoint = np.fromfile(fid, np.float32, 1)[0] / 1000.0 # in microseconds per channel, divided by 1000 to msec
            self.Protocol.numPoints = np.fromfile(fid, np.int32, 1)[0] # number of data points
            self.Protocol.WCtime = np.fromfile(fid, np.float32, 1)[0] # in seconds since went whole cell
            self.Protocol.WCtimeStr = self.epiTime(self.Protocol.WCtime)
            self.Protocol.drugTime = np.fromfile(fid, np.float32,1)[0] # in seconds since most recent drug started
            self.Protocol.drugTimeStr = self.epiTime(self.Protocol.drugTime)
            self.Protocol.drug = int(np.fromfile(fid,np.float32,1)[0]) #an integer indicating what drug is on

            #% new from BWS on 12/21/08
            np.fromfile(fid,np.int32,1) # simulated data
            fid.seek(48 , 0)
            self.Protocol.genData = np.fromfile(fid, np.float32, 56) # [need expansion]

            # read in TTL information
            self.Protocol.ttlData = []
            ttlDataStr = ""
            chanCounter = 0
            for index in range(self.Protocol.numChannels):
                fid.seek(10, 1) # 10 is for VB user-defined type stuff
                self.Protocol.ttlData.append(np.fromfile(fid, np.float32, 17)) #[need expansion]
                ttlDataStr += self.generateTTLdesc(chanCounter, self.Protocol.ttlData[-1])
                chanCounter += 1
            #print(fid.tell())
            
            self.Protocol.ttlDict = []
            for index, ttlData in enumerate(self.Protocol.ttlData):
                self.Protocol.ttlDict.append(self.parseTTLArray_old(ttlData))
                
            # read in DAC information
            self.Protocol.dacData = []
            self.Protocol.dacName = []
            dacDataStr = ""
            chanCounter = 0
            for index in range(self.Protocol.numChannels):
                fid.seek(10, 1) # 10 is for VB user-defined type stuff
                self.Protocol.dacData.append(np.fromfile(fid, np.float32, 42)) #[need exspansion]
                self.Protocol.dacName.append(readVBString(fid))
                dacDataStr += self.generateDACdesc(chanCounter, self.Protocol.dacData[-1])
                chanCounter += 1

            #print(fid.tell())
            # Get other parameters
            self.Protocol.classVersionNum = np.fromfile(fid, np.float32, 1)[0]
            self.Protocol.acquireComment=readVBString(fid)
            self.Protocol.acquireAnalysisComment=readVBString(fid)
            self.Protocol.drugName=readVBString(fid)
            self.Protocol.exptDesc=readVBString(fid)
            self.Protocol.computerName=readVBString(fid)
            self.Protocol.savedFileName=os.path.abspath(readVBString(fid)).replace('\\','/')
            self.Protocol.fileName = self.Protocol.savedFileName
            self.Protocol.linkedFileName=os.path.abspath(readVBString(fid)).replace('\\','/')
            self.Protocol.acquisitionDeviceName=readVBString(fid)
            self.Protocol.traceKeys=readVBString(fid)
            self.Protocol.traceInitValuesStr=readVBString(fid)
            self.Protocol.extraScalarKeys=readVBString(fid)
            self.Protocol.extraVectorKeys=readVBString(fid)
            self.Protocol.genString=readVBString(fid)
            self.Protocol.TTLstring = []
            for index in range(self.Protocol.numChannels):
                self.Protocol.TTLstring.append(readVBString(fid))
            self.Protocol.ampDesc = []
            for index in range(self.Protocol.numChannels):
                self.Protocol.ampDesc.append(readVBString(fid))

            # Stimulus description
            self.Protocol.stimDesc = (dacDataStr.strip() + " " + ttlDataStr.strip() + " " + self.Protocol.acquireComment).strip()

            # Get Channel info
            channelDict = {'VoltADC1':'VoltA','VoltADC3':'VoltB',
                           'VoltADC5':'VoltC','VoltADC7':'VoltD',
                           'CurADC0':'CurA','CurADC2':'CurB',
                           'CurADC4':'CurC','CurADC6':'CurD',
                           'StimulusAmpA':'StimA',
                           'StimulusAmpB':'StimB',
                           'StimulusAmpC':'StimC',
                           'StimulusAmpD':'StimD',
                           'StimulusAmpA9':'StimA',
                           'StimulusAmpB9':'StimB',
                           'StimulusAmpC9':'StimC',
                           'StimulusAmpD9':'StimD'}
            keys = [k.split("/")[0] for k in self.Protocol.traceKeys.split()]
            self.Protocol.channelNames = [channelDict[k] for k in keys]
            self.Protocol.numTraces = len(self.Protocol.channelNames)

            if infoOnly: # stop here if only
                return

            # print(fid.tell())
            # Read trace data
            self.Protocol.traceDesc = []
            for chan in self.Protocol.channelNames:
                traceFactor = float(np.fromfile(fid, np.float32, 1))
                traceLength = int(np.fromfile(fid, np.int32, 1))
                traceDesc = readVBString(fid)
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

            if getTime:
                self.Time = np.arange(self.Protocol.numPoints) * self.Protocol.msPerPoint

        # close file
        fid.close()

    def TTL2Stim_old(self, TTLarray):
        TTL = self.parseTTLArray_old(TTLarray)
        TTL_trace = np.arange(0, duration+ts, ts)
        return TTL_trace
        
        
    @staticmethod
    def epiTime(inTime):
        """Convert seconds into HH:MM:SS"""
        if inTime>=3600:
            hh = int(inTime//3600)
            mm = int((inTime - hh*3600)//60)
            ss = inTime - hh*3600 - mm*60
            return "{:0d}:{:02d}:{:0.0f}".format(hh, mm, ss)
        elif inTime>=60:
            mm = int(inTime // 60)
            ss = inTime - mm *60
            return "{:0d}:{:02.0f}".format(mm, ss)
        else:
            return("{:0.1f} sec".format(inTime))

    @staticmethod
    def generateDACdesc(chanNum, data):
        # revised 13 July 2015 BWS
        step = ""
        pulse = ""
        result = ""
        if data[0]:
            if data[1] and data[2]:
                step = "Step " + "{:0.0f}".format(data[8]) + " (" + "{:0.0f}".format(data[6]) + " to " + \
                    "{:0.0f}".format(data[7]) + " ms)"
            if data[14]:
                if data[17] != 0:
                    pulse += "PulseA " + "{:0.0f}".format(data[17]) + " "
                if data[20] !=  0:
                    pulse += "PulseB " + "{:0.0f}".format(data[20]) + " "
                if data[23] != 0:
                    pulse += "PulseC " + "{:0.0f}".format(data[23]) + " "
            if len(step) > 0 or len(pulse) > 0:
                result = "DAC" + str(chanNum) + ": "
                if len(step) > 0:
                    result += step.strip() + " "
                if len(pulse) > 0:
                    result += pulse.strip()
        return result.strip()

    @staticmethod
    def generateTTLdesc(chanNum, data):
        # revised 13 July 2015 BWS
        SIU = None
        Puff = None
        if data[0]: # global enable
            tempStr = ""
            if data[5]: # single SIU enable
                for k in range(6,10):
                    if data[k] > 0.:
                        tempStr += str(data[k]) + " ms "
            if data[10]: # SIU train enable
                tempStr += " train"
            if len(tempStr) > 0:
                SIU = "SIU " + tempStr
            tempStr = ""
            if data[2]: # TTL step enable
                Puff = "Puff " + str(data[4]) + "ms"
        if SIU or Puff:
            retStr = "TTL" + str(chanNum) + ": "
            if Puff:
                retStr += Puff + " "
            if SIU:
                retStr += SIU
        else:
            retStr = ""
        return retStr.strip()
    
    @staticmethod
    def parseTTLArray_old(TTLarray):
        """Convert the TTL array into meaningful dictionary"""
        TTL = OrderedDict()
        TTL['is_on'] = TTLarray[0]
        TTL['use_AWF'] = TTLarray[1]
        TTL['Step_is_on'] = TTLarray[2]
        TTL['Step_Latency'] = TTLarray[3]
        TTL['Step_Duration'] = TTLarray[4]
        TTL['SIU_Single_Shocks_is_on'] = TTLarray[5]
        TTL['SIU_A'] = TTLarray[6]
        TTL['SIU_B'] = TTLarray[7]
        TTL['SIU_C'] = TTLarray[8]
        TTL['SIU_D'] = TTLarray[9]
        TTL['SIU_Train_is_on'] = TTLarray[10]
        TTL['SIU_Train_of_Bursts_is_on'] = TTLarray[11]
        TTL['SIU_Train_Start'] = TTLarray[12]
        TTL['SIU_Train_Interval'] = TTLarray[13] # stimulate every x ms
        TTL['SIU_Train_Number'] = TTLarray[14]
        TTL['SIU_Train_Burst_Interval'] = TTLarray[15]
        TTL['SIU_Train_Burst_Number'] = TTLarray[16]
        return TTL

    @staticmethod
    def parseGenArray_old(GenArray):
        """Convert genData array into meaningful dictionary"""
        Gen = OrderedDict()
        Gen['chantype'] = GenArray[3:11]
        Gen['chanGain'] = GenArray[11:19]
        Gen['chanExtGain'] = GenArray[19:27]
        Gen['AuxTTlEnable'] = GenArray[51]
        Gen['extTrig'] = GenArray[52]
        Gen['SIUDuration'] = GenArray[53]
        Gen['episodicMode'] = GenArray[54]
        Gen['programCode'] = GenArray[55]
        return Gen
        

def load_trace(cellname, basedir='D:/Data/Traces', old=True, infoOnly=False, *args, **kwargs):
    """Wrapper function to load NeuroData, assuming the data structure we have
    implemented in get_cellpath"""
    cell_path = os.path.join(basedir, get_cellpath(cellname))
    zData = NeuroData(dataFile=cell_path, old=old, infoOnly=infoOnly, *args, **kwargs)
    return zData

""" 2photon image data"""
class ImageData(object):
    """Read image data file
    """
    def __init__(self, dataFile=None, old=False, *args, **kwargs):
        """Initialize class"""
        self.img = None
        self.Protocol = Protocol()

        if dataFile is not None and isinstance(dataFile, str):
            # load directly if all the conditions are met
            self.LoadData(dataFile=dataFile, old=old, *args, **kwargs)
        else:
            raise(IOError('Unrecognized data file input'))

    def LoadData(self, dataFile, old=True, *args, **kwargs): #old=True to be edited later
        """Load data in text file"""
        dataFile = dataFile.replace('\\','/')# make sure using forward slash
        # check file exists
        if not os.path.isfile(dataFile):
            raise(IOError('%s does not exist' %dataFile))
        # Evoke proper load method
        if old:
            self.LoadOldDataFile(dataFile, *args, **kwargs)
        else:
            self.LoadDataFile(dataFile, *args, **kwargs)

    def LoadDataFile(self, dataFile, infoOnly=False):
        raise(NotImplementedError("Cannot load new data format yet"))

    def LoadOldDataFile(self, dataFile, infoOnly=False):
        """ Read a .img file"""
        fid = open(dataFile, 'rb')
        self.Protocol.FileName = dataFile
        self.Protocol.BitDepth = 12
        self.Protocol.ProgramNumber = np.fromfile(fid, np.int32, 1)
        if self.Protocol.ProgramNumber == 2:
            fid.close()
            self.loadQuantixFile(dataFile)
            return

        self.Protocol.ProgramMode = np.fromfile(fid, np.int32, 1)[0]
        self.Protocol.DataOffset = np.fromfile(fid, np.int32, 1)[0]
        self.Protocol.Width = np.fromfile(fid, np.int32, 1)[0]
        self.Protocol.Height = np.fromfile(fid, np.int32, 1)[0]
        self.Protocol.NumImages = np.fromfile(fid, np.int32, 1)[0]
        self.Protocol.NumChannels = np.fromfile(fid, np.int32, 1)[0]
        self.Protocol.Comment = readVBString(fid)
        self.Protocol.MiscInfo = readVBString(fid)
        self.Protocol.ImageSource = readVBString(fid)
        self.Protocol.PixelMicrons = np.fromfile(fid, np.float32, 1)[0]
        self.Protocol.MillisecondPerFrame = np.fromfile(fid, np.float32, 1)[0]
        self.Protocol.Objective = readVBString(fid)
        self.Protocol.AdditionalInformation = readVBString(fid)
        self.Protocol.SizeOnSource = readVBString(fid)
        self.Protocol.SourceProcessing = readVBString(fid)

        # fix calibration parameters
        if not self.Protocol.PixelMicrons or self.Protocol.PixelMicrons == 0:
            if self.Protocol.Objective.upper() == 'OLYMPUS 60X/0.9':
                self.Protocol.PixelMicrons = 103.8 / float(re.match('Zoom = (\d+)', self.Protocol.SourceProcessing, re.M|re.I).group(1)) / self.Protocol.Width
            elif self.Protocol.Objective.upper() == 'OLYMPUS 40X/0.8':
                self.Protocol.PixelMicrons = 163 / float(re.match('Zoom = (\d+)', self.Protocol.SourceProcessing, re.M|re.I).group(1)) / self.Protocol.Width

        self.Protocol.Origin = []
        for n, c in enumerate(['X','Y','Z']):
            coord = r"(?<=" + c + r" = )[\d.-]+"
            coord = re.search(coord, self.Protocol.MiscInfo)
            if coord:
                self.Protocol.Origin.append(float(coord.group(0)))
            else:
                self.Protocol.Origin.append(None)

        self.Protocol.delta = [self.Protocol.PixelMicrons, self.Protocol.PixelMicrons, self.Protocol.PixelMicrons]

        # information for convenience
        self.Xpixels = self.Protocol.Width
        self.Ypixels = self.Protocol.Height
        self.numChannels = self.Protocol.NumChannels
        self.numFrames = self.Protocol.NumImages

        if not infoOnly:
            # read image data
            fid.seek(self.Protocol.DataOffset - 1, 0)
            self.img = np.zeros((self.Protocol.Height, self.Protocol.Width, self.Protocol.NumImages), dtype=np.int16)
            # set_trace()
            for x in range(self.Protocol.NumImages):
                tmp = np.fromfile(fid, np.int16, self.Protocol.Width * self.Protocol.Height) # / 4096
                self.img[:,:,x] = tmp.reshape((self.Protocol.Width, self.Protocol.Height), order='F').T
            # Rotate the image
            # self.img = self.img[:, ::-1, :]
        else:
            self.img = -1

        fid.close()

    def loadQuantixFile(self, dataFile, infoOnly=False):
        raise(NotImplementedError('Function to load Quantix File not implemented'))


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

    def Neuro2Trace(self, data, channels=None, streams=None, protocol=False,
                    *args, **kwargs):
        """Use NeuroData method to load and parse trace data to be plotted
        data: an instance of NeuroData, ro a list of instances
        channels: list of channels to plot, e.g. ['A','C','D']
        streams: list of data streams, e.g. ['V','C','S']
        protocol: load protocol to meta data. Default False.
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
        if protocol:
            self.meta.update({'protocol':[]})
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
            if protocol:
                self.meta['protocol'].append(d.Protocol)
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

class FormatException(Exception):
    def __init___(self,dErrorArguments):
        print(dErrorArguments)
        Exception.__init__(self,"Invalid cell label {0}. Format: Name.ddMMMyy, e.g. Cell A.02Jun15".format(dErrorArguments))
        self.dErrorArguments = dErrorArguments

# Implement a simple data loader with the assumption of current data structure
def get_cellpath(cell_label, episode='.{}', year_prefix='20'):
    """Infer full path of the cell given cell label (without file extension)
    e.g. Neocortex A.09Sep15.S1.E13 should yield
    ./2015/09.September/Data 9 Sep 15/Neocortex A.09sep15.S1.E13.dat"""
    cell_label = cell_label.strip('.dat')

    if episode[0] != '.':
        episode = '.'+episode

    dinfo = re.findall('([\w\s]+).(\d+)([a-z_A-Z]+)(\d+).S(\d+).E(\d+)', cell_label)

    if not dinfo:
        dinfo = re.findall('([\w\s]+).(\d+)([a-z_A-Z]+)(\d+)', cell_label)
    else:
        episode = ''

    try:
        dinfo = dinfo[0]
    except:
        raise(FormatException("Invalid cell label {0}. Format: Name.ddMMMyy, e.g. Cell A.02Jun15".format(cell_label)))

    # year folder
    year_dir = year_prefix + dinfo[3]
    # month folder
    month_dict = {'Jan':'01.January','Feb':'02.February','Mar':'03.March',\
    'Apr':'04.April','May':'05.May', 'Jun':'06.June', 'Jul':'07.July',\
    'Aug':'08.August','Sep':'09.September','Oct':'10.October',\
    'Nov':'11.November','Dec':'12.December'}
    month_dir = month_dict[dinfo[2]]
    # data folder
    data_folder = "Data {:d} {} {}".format(int(dinfo[1]),  dinfo[2], year_dir)
    data_folder = os.path.join(year_dir, month_dir, data_folder, cell_label+episode+'.dat')
    data_folder = data_folder.replace('\\','/')
    return data_folder


class ROI(object):
    """Helper class for structuring ROIs"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class ROIData(list):
    def __init__(self, roifile=None, old=True, *args, **kwargs):
        self.roifile = roifile
        if roifile is not None:
            self.loadROI(roifile=roifile, old=old, *args, **kwargs)

    def loadROI(self, roifile, old=True, *args, **kwargs):
        if old:
            self.loadOldROIData(roifile, *args, **kwargs)

    def loadOldROIData(self, roifile, roitype='square'):
        fid = open(roifile, 'rb')
        fid.seek(4, 0)
        n = 0
        while n < 1000:
            # initialize
            roi = ROI(center=0, unknown1=0, size=0, unknown2=0, position=0)
            roi.center = np.fromfile(fid, np.int16, 2)
            if isempty(roi.center):
                return
            roi.unknown1 = np.fromfile(fid, np.int16, 1)
            roi.size = np.fromfile(fid, np.int16, 2)
            roi.unknwon2 = np.fromfile(fid, np.int16, 9)
            # Position of the square
            roi.position = np.fromfile(fid, np.int16, 4)
            roi.position = np.reshape(roi.position, (2,2))
            self.append(roi)
            n += 1

        # should in theory never reach this, but could indicate some problem
        if n>=1000:
            print('Maximum iteration exceeded. Loading only 1000 ROIs')



if __name__ == '__main__' and True:
#    data = NeuroData(dataFile, old=True)
#    figdata = FigureData()
    # dataFile = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/example/lineplot.csv'
#    figdata.Neuro2Trace(data, channels=['A','B','C','D'], streams=['Volt','Cur','Stim'])
    # data = FigureData(dataFile='D:/Data/2015/07.July/Data 2 Jul 2015/Neocortex C.02Jul15.S1.E40.dat',old=True, channels=['A'], streams=['Volt','Cur','Stim'])
    zData = NeuroData(dataFile='D:/Data/Traces/2015/07.July/Data 13 Jul 2015/Neocortex I.13Jul15.S1.E7.dat', old=True, infoOnly=True)
    # mData = ImageData(dataFile = 'D:/Data/2photon/2015/03.March/Image 10 Mar 2015/Neocortex D/Neocortex D 01/Neocortex D 01.512x512y1F.m21.img', old=True)
    # plt.imshow(mData.img[:,:,0])
    # zData = load_trace('Neocortex F.15Jun15.S1.E10')
    #roifile = 'C:/Users/Edward/Desktop/Slice B CCh Double.512x200y75F.m1.img.roi'
    #R = ROIData(roifile)
