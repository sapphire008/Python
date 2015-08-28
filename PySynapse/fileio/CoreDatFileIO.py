# -*- coding: utf-8 -*-

# last revised 13 July 2015 BWS  working for B and C type of DAT files

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from numpy import asscalar, fromfile, mean, float64, float32, int16, int32, uint8
import pyperclip as pclip
import pickle as pck
import os
import sys
import ntpath as path
import shutil
import scipy.io as sio
import glob
import os.path as path
import numpy as np

def readDatFile(fileName,  readTraceData = True):
    if not path.isfile(fileName):
        print("bad file for readDatFile")
        return None
    statinfo = os.stat(fileName)
    if statinfo.st_size == 0:
        print("skipping zero byte file: " + path.basename(fileName))
        return None
    f = open(fileName, "rb")
    protocolType = asscalar(fromfile(f, int16, 1))
    acquireVersion = asscalar(fromfile(f, int32, 1))
    f.close()
    if protocolType == 25.0:
        if acquireVersion < 8:
            mEpi = readDatFileTypeB(fileName, readTraceData)
        else:
            mEpi = readDatFileTypeC(fileName, readTraceData) 
        return mEpi
    else:
        print("Unknown data file type: " + fileName)
        return None

def readCommonDatBeginning(fileName, f):
    mEpi = {}
    mEpi["loadedFileName"] = fileName
    nameParts = fileName.split(".")
    mEpi["shortNameCode"] = nameParts[2] + "." + nameParts[3]
    mEpi["protocolType"] = asscalar(fromfile(f, int16, 1))
    mEpi["acquireVersion"] = asscalar(fromfile(f, int32, 1))
    infoBytes = asscalar(fromfile(f, int32, 1))
    sweepWindow = asscalar(fromfile(f, float32, 1))
    mEpi["sweepWindow"] = sweepWindow
    mEpi["msPerPoint"] = asscalar(fromfile(f, float32, 1)) / 1000
    mEpi["numPoints"] = asscalar(fromfile(f, int32, 1))
    cellTime = asscalar(fromfile(f, float32, 1))
    mEpi["cellTime"] = cellTime
    mEpi["cellTimeStr"] = formatTime(cellTime)
    drugTime = asscalar(fromfile(f, float32, 1))
    mEpi["drugTime"] = drugTime
    mEpi["drugTimeStr"] = formatTime(drugTime)
    drugLevel = asscalar(fromfile(f, float32, 1))
    mEpi["drugLevel"] = drugLevel
    mEpi["drugLevelStr"] = "DL" + str(int(drugLevel))
    mEpi["simulatedData"] = asscalar(fromfile(f, int32, 1))
    f.seek(48)
    genData = (fromfile(f, float32, 56)).tolist()
    if genData[31] == 4:
        mEpi["clampModeStr"] = "VC"
    else:
        mEpi["clampModeStr"] = "CC"
    mEpi["genData"] = genData
    ttlData = [];
    ttlDesc = ""
    for chan in range(1,5):
        f.seek(f.tell() + 10)
        ttlDataTemp = (fromfile(f, float32, 17))
        ttlData.append(ttlDataTemp)
        ttlDesc += generateTTLdesc(chan - 1, ttlDataTemp) + " "
        mEpi["ttlData" + str(chan-1)] = ttlData
    dacData = []
    dacStrings = []
    dacDesc = ""
    for chan in range(1,5):
        f.seek(f.tell() + 10)
        dacDataTemp = (fromfile(f, float32, 42)).tolist()
        dacData.append(dacDataTemp)
        dacDesc += generateDACdesc(chan - 1, dacDataTemp) + " "
        tempLength = asscalar(fromfile(f, int16, 1))
        if tempLength > 0:
            dacStringTemp = arrayToString(fromfile(f, uint8, tempLength))
            dacStrings.append(dacStringTemp)
        else:
            dacStrings.append("no DAC string")
    mEpi["dacData"] = dacData
    mEpi["mainDacStr"] = dacDesc
    mEpi["dacStrings"] = dacStrings
    tempStr = dacDesc.strip() + " " + ttlDesc.strip()
    if len(tempStr) == 0:
        tempStr = "no desc"
    mEpi["stimDesc"] = tempStr.strip()
    return mEpi, infoBytes

def binaryFileToString(f):
    tempLength = asscalar(fromfile(f, int16, 1))
    if tempLength > 0:
        return arrayToString(fromfile(f, uint8, tempLength))
    else:
        return "no string"

def readDatFileTypeC(fileName, readTraceData = True):
    f = open(fileName, "rb")
    mEpi, infoBytes = readCommonDatBeginning(fileName, f)
    mEpi["numPoints"] = mEpi["numPoints"] - 1
    mEpi["fileVersion"] = "C"
    mEpi["classVersion"] = asscalar(fromfile(f, float32, 1))
    mEpi["acquireComment"] = binaryFileToString(f)
    mEpi["acquireAnalysisComment"] = binaryFileToString(f)
    mEpi["drugName"] = binaryFileToString(f)
    mEpi["exptDesc"] = binaryFileToString(f)
    mEpi["computerName"] = binaryFileToString(f)
    mEpi["savedFileName"] = binaryFileToString(f)
    mEpi["linkedFileName"] = binaryFileToString(f)
    mEpi["acquisitionDeviceName"] = binaryFileToString(f)
    mEpi["traceKeysAsOneString"] = binaryFileToString(f)
    mEpi["chanNames"] = fixKeyVerC(mEpi["traceKeysAsOneString"])
    mEpi["traceInitValuesStr"] = binaryFileToString(f)
    mEpi["extraScalarKeys"] = binaryFileToString(f)
    mEpi["extraVectorKeys"] = binaryFileToString(f)
    mEpi["genStr"] = binaryFileToString(f)
    for ii in range(4):
        mEpi["TTLstring" + str(ii)] = binaryFileToString(f)
    for ii in range(4):
        mEpi["AmpDesc" + str(ii)] = binaryFileToString(f)
    if readTraceData:
        mTraces = {}
        mInitValues = {}
        numBaselinePoints = 49
        for oneChanName in mEpi["chanNames"]:
            traceFactor = asscalar(fromfile(f, float32, 1))
            traceLength = asscalar(fromfile(f, int32, 1))
            traceDesc = binaryFileToString(f)
            tempChan = fromfile(f, int16, traceLength)
            mTraces[oneChanName] = traceFactor * tempChan[0:-1]
            mInitValues[oneChanName] = mean(mTraces[oneChanName][0:numBaselinePoints])
        mEpi["traces"] = mTraces
        mEpi["initValues"] = mInitValues
    f.close()
    return mEpi

def readDatFileTypeB(fileName, readTraceData = True):
    # ToDo: Dac-then-TTLdescStr TryCatchErrorMsg
    f = open(fileName, "rb")
    mEpi, infoBytes = readCommonDatBeginning(fileName, f)
    mEpi["fileVersion"] = "B"
    mEpi["statBoxes"] = (fromfile(f, int32, 25)).tolist()
    mEpi["statValues"] = (fromfile(f, float32, 25)).tolist()
    statNames = []
    for box in range(1,26):
        tempLength = asscalar(fromfile(f, int16, 1))
        if tempLength > 0:
            statStringTemp = arrayToString(fromfile(f, uint8, tempLength))
            statNames.append(statStringTemp)
        else:
            statNames.append("no stat string")
    mEpi["statNames"] = statNames
    tempLength = asscalar(fromfile(f, int16, 1))
    mEpi["comment"] = arrayToString(fromfile(f, uint8, tempLength))
    tempLength = asscalar(fromfile(f, int16, 1))
    mEpi["analysisComment"] = arrayToString(fromfile(f, uint8, tempLength))
    tempLength = asscalar(fromfile(f, int16, 1))
    mEpi["savedFileName"] = arrayToString(fromfile(f, uint8, tempLength))
    chanNames = []
    numTraces = 0
    for box in range(1,21):
        tempLength = asscalar(fromfile(f, int16, 1))
        if tempLength > 0:
            chanStringTemp = arrayToString(fromfile(f, uint8, tempLength))
            chanNames.append(convertOldToNewKeyVerB(chanStringTemp))
            numTraces += 1
    mEpi["chanNames"] = chanNames
    mEpi["numTraces"] = numTraces
    mInitValues = {}
    numReadPoints = mEpi["numPoints"] + 1
    numBaselinePoints = 49
    f.seek(infoBytes - 1)
    for chan in mEpi["chanNames"]:
        tempChan = fromfile(f, float32, numBaselinePoints).astype(float64)
        mInitValues[chan] = mean(tempChan)
        f.seek(f.tell() + (4 * (numReadPoints - numBaselinePoints)))
    mEpi["initValues"] = mInitValues
    if readTraceData:
        mTraces = {}
        f.seek(infoBytes - 1)
        for chan in mEpi["chanNames"]:
            tempChan = fromfile(f, float32, numReadPoints)
            mTraces[chan] = tempChan.astype(float64)[0:-1]
        mEpi["traces"] = mTraces
    f.close()
    return mEpi

def generateEpiDescList(fileNameWildCard):
    # revised 21 July 2013 BWS
    fList = glob.glob(fileNameWildCard)
    outList = fList.copy()
    count = 0
    for epiName in fList:
        epi = readDatFile(epiName, False)
        outList[count] = epi["shortNameCode"] + " | " + epi["traceDesc"] + " | " + epiName
        count += 1
    return outList

def clearScratchFolder(scratchFolder):
    # revised 21 July 2013 BWS
    if path.exists(scratchFolder):
        shutil.rmtree(scratchFolder)

def doesEpiFolderExist(epiFileName, scratchPath):
    # revised 21 July 2013 BWS
    root = path.splitext(path.basename(epiFileName))[0]
    dataPath = scratchPath + "\\" + root + "\\"
    return path.exists(dataPath)

def dumpEpisodeInFolder(epi, scratchPath):
    # revised 9 June 2015 BWS
    # create clean subfolder
    root = path.splitext(path.basename(epi["loadedFileName"]))[0]
    dataPath = scratchPath +"/" + root + "/"
    if path.exists(dataPath):
        shutil.rmtree(dataPath)
    os.makedirs(dataPath)
    # write info text file
    txtFile = dataPath + "fileInfo.txt"
    f1 = open(txtFile, "w")
    print("fileName = " + epi["loadedFileName"], file=f1)
    print("stimDesc = " + epi["stimDesc"], file=f1)
    if "traceDesc" in epi:
        print("traceDesc = " + epi["traceDesc"], file=f1)
    print("msPerPoint = " + "{:0.1f}".format(epi["msPerPoint"]), file=f1)
    print("sweepWindow = " + "{:0.0f}".format(epi["sweepWindow"]), file=f1)
    if "comment" in epi:
        print("comment = " + epi["comment"], file=f1)
    print("acquiredTraces = " + str(epi["chanNames"]), file=f1)
    f1.close()
    del f1
    # dump traces as raw float64 binary vectors if they are in episode dict
    if "traces" in epi.keys():
        for traceName in epi["traces"].keys():
            (epi["traces"][traceName]).tofile(dataPath + traceName + ".dat")
        print("Saved episode traces in " + dataPath)
    # dump pickeled version of header last since we wipe out any trace data
    matFile = dataPath + "protocol.mat"
    pckFile = dataPath + "header.pkc"
    f2 = open(pckFile, "wb")
    if "traces" in epi.keys():
        tempTraces = epi["traces"]
        del epi["traces"]
        pck.dump(epi, f2)
        sio.savemat(matFile, epi, oned_as='row')
        epi["traces"] = tempTraces
        del tempTraces
    else:
        pck.dump(epi, f2)
        sio.savemat(matFile, epi, oned_as='row')
    f2.close()
    del f2

def putVector(newVector, vecName, pathEndHint = "0"):
    exactPath = resolvePathEndHint(pathEndHint)
    if exactPath:
        if (len(vecName) > 0) & (size(newVector) > 0):
            if vecName.upper()[-4:] != ".DAT":
                vecName += ".dat"
            fullName = exactPath + "\\" + vecName
            fullName = xpath.normpath(fullName).strip() # boilerplate to clean up path problems
            newVector.tofile(fullName)
            return True
    return False
def getVector(vecHint, pathEndHint = "0"):
    exactPath = resolvePathEndHint(pathEndHint)
    if exactPath:
        return getVectorExactPath(vecHint, exactPath)
    return None
def getVectorAverage(vecHint):
    realVec = resolveVectorHint(vecHint)
    if realVec == None:
        return None
    count = 0
    for folder in getDataFolders():
        if count == 0:
            traceAverage = getVectorExactPath(realVec, folder)
            if traceAverage == None:
                return None
        else:
            newTrace = getVectorExactPath(realVec, folder)
            if newTrace == None:
                print("Did not find requested vector in all data folders: " + realVec)
                return None
            else:
                traceAverage = traceAverage + newTrace
        count += 1
    if count > 0:
        return np.divide(traceAverage, count)
    else:
        print("No traces found to average")
        return None
def getProtocol(pathEndHint = "0"):
    exactPath = resolvePathEndHint(pathEndHint)
    if exactPath:
        protocolFile = exactPath + "\\" + "header.pkc"
        if xpath.isfile(protocolFile):
            f3 = open(protocolFile, "rb")
            protocol = pck.load(f3)
            f3.close()
            del f3
            return protocol
    return None
def getProtocolValue(keyStr, pathEndHint = "0"):
    p = getProtocol(pathEndHint)
    if p:
        if keyStr in p:
            return p[keyStr]
        else:
            print("Count not find key in header: " + keyStr)
    return None

#
# Internal functions below here
#

def getVectorExactPath(vecHint, dataFolder):
    exactVecName = resolveVectorHint(vecHint, dataFolder)
    if exactVecName:
        return getVectorExact(exactVecName, dataFolder)
    else:
        return None
def getVectorExact(vecName, dataFolder):
    if vecName.upper()[-4:] != ".DAT":
        vecName += ".dat"
    fullName = dataFolder + "\\" + vecName
    fullName = xpath.normpath(fullName).strip() # boilerplate to clean up path problems
    if xpath.isfile(fullName):
        return np.fromfile(fullName, np.float64)
    else:
        print("Problem loading vector: " + fullName)
        return None
def getDataFolders(scratchFolder = "r:\\Synapse"):
    retList = []
    if xpath.isdir(scratchFolder):
        for fName in os.listdir(scratchFolder):
            tempPath = scratchFolder + "\\" + fName
            if xpath.isdir(tempPath):
                retList.append(tempPath)
    return retList
def getTestValue():
    return 72
def getVectorNames(pathEndHint = "0"):
    curFolder = resolvePathEndHint(pathEndHint)
    retList = []
    for fName in os.listdir(curFolder):
        if fName.upper()[-4:] == ".DAT":
            retList.append(fName[:-4])
    return retList
def resolveVectorHint(vecHint, exactDataPath = ""):
    if len(vecHint) == 0:
        print("Must provide some vector hint string")
        return None
    if exactDataPath == "":
        fList = getDataFolders()
        if fList:
            exactDataPath = fList[0]
        else:
            print("Could not find any data folders")
            return None
    vNames = getVectorNames(exactDataPath)
    if vecHint in vNames:
        return vecHint
    else:
        testString = vecHint.upper()
        testLength = len(testString)
        for possibleVec in vNames:
            pVecU = possibleVec.upper()
            if pVecU[:testLength] == testString:
                return possibleVec
        print("Could not resolve vector hint: " + vecHint + " from " + str(vNames))
        return None
def resolvePathEndHint(pathEndHint):
    allDataFolders = getDataFolders()
    if len(allDataFolders) == 0:
        print("No data folders exist")
        return None
    if (pathEndHint == "0") | (len(pathEndHint) == 0):
        curDataFolder = allDataFolders[0] # just get first data folder
    else:
        curDataFolder = ""
        testString = pathEndHint.upper()
        testLength = -1 * len(pathEndHint)
        for testFolder in allDataFolders:
            testFolderU = testFolder.upper()
            if testFolderU[testLength:] == testString:
                curDataFolder = testFolder
                break
        if len(curDataFolder) == 0:
            print("Could not match existing data folders with path end hint: " + pathEndHint)
            return None
    return curDataFolder


def generateTraceDesc(epi):
    # revised 21 July 2013 BWS
    traceDesc =formatTime(epi["cellTime"])
    traceDesc += " (" + "{:0.0f}".format(epi["sweepWindow"] / 1000) + " sec) "
    traceDesc += " " + epi["clampModeStr"] + " "
    if epi["drugLevel"] > 0:
        traceDesc += " DL0"
    else:
        traceDesc += " DL" + "{:0.0f}".format(epi["drugLevel"]) # xx add drugTime in parens
    traceDesc += " " + epi["stimDesc"]
    return traceDesc.strip()

def mapOldToNewChan(oldChan):
    if ord(oldChan) >= 65:
        return oldChan
    else:
        oldNum = int(oldChan)
        if oldNum in [0,1]:
            return "A"
        elif oldNum in [2,3]:
            return "B"
        elif oldNum in [4,5]:
            return "C"
        elif oldNum in [6,7]:
            return "D"
        else:
            return "X"

def convertOldToNewKeyVerB(oldKey):
    if oldKey[0:3] == "Vol":
        newKey = "Volt" + mapOldToNewChan(oldKey[8])
    elif oldKey[0:3] == "Cur":
        newKey = "Cur" + mapOldToNewChan(oldKey[7])
    elif oldKey[0:3] == "Sti":
        newKey = "Stim" + mapOldToNewChan(oldKey[13])
    else:
        newKey = "Unknown"
    return newKey

def convertOldToNewKeyVerC(oldKey):
    if oldKey[0:3] == "Vol":
        newKey = "Volt" + mapOldToNewChan(oldKey[7])
    elif oldKey[0:3] == "Cur":
        newKey = "Cur" + mapOldToNewChan(oldKey[6])
    elif oldKey[0:3] == "Sti":
        #print(oldKey[0:11] + str(oldKey[0:11] == "StimulusAmp"))
        if oldKey[0:11] == "StimulusAmp":
            newKey = "Stim" + oldKey[11]
        else:
            newKey = "Stim" + mapOldToNewChan(oldKey[12])
    else:
        newKey = "Unknown"
    return newKey

def fixKeyVerC(keyListStr):
    keys = keyListStr.split(" ")
    for ii in range(len(keys)):
        keys[ii] = convertOldToNewKeyVerC(keys[ii])
    return keys

def arrayToString(inArray):
    outString = ""
    if len(inArray) > 0:
        for i in arrayToStringGenerator(inArray):
            outString += i
        return outString
    else:
        return ""

def arrayToStringGenerator(passArray):
    i = 0
    while i < len(passArray):
        yield chr(passArray[i])
        i += 1

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

def formatTime(inTime):
    # revised 20 July 2013 BWS
    f0 = "{:0.0f}"
    f1 = "{:0.1f}"
    d2 = "{:2.0f}"
    tempS = ""
    inTime += 0.1
    if inTime <= 59.9:
        tempS = f1.format(inTime) + " sec"
    else:
        if inTime <= 3599.9:
            M = inTime // 60
            S = inTime - (M * 60)
            tempS = f0.format(M) + ":" + d2.format(S).strip().zfill(2)
        else:
            H = inTime // 3600
            temp = inTime - (H * 3600)
            M = temp // 60
            S = temp - (M * 60)
            tempS = f0.format(H) + ":" + d2.format(M).strip().zfill(2) + ":" + d2.format(S).strip().zfill(2)
    return tempS

if __name__ == "__main__":
    if len(sys.argv) == 3:
        localEpi = readDatFile(sys.argv[1])
        if localEpi:
            dumpEpisodeInFolder(localEpi, sys.argv[2])
        else:
            print("Problem creating episode dict")
    else:
        print("This function requires two arguments: fileNameWithPath and scratchFolder")
