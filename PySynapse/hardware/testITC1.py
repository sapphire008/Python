import ITC as mITC

fN = 'D:/LabWorld/Executables/ITCMM64.dll'
ITC = mITC.PyITC(fN)

itcParm = {}
itcParm["saveFileName"] = None
itcParm["msPerPoint"] = 0.2
itcParm["sweepWindowMs"] = 500
itcParm["activeADCs"] = [0, 1]
itcParm["activeDACs"] = [0]
itcParm["ADCfullScale"] = [10, 10]
itcParm["ADCnames"] = ["CurA", "VoltA"]
itcParm["ADCmultiplyFactors"] = [5.1, 0.123]
stimDict = {}
stimDict["DAC0"] = "step 100 350 100"
itcParm["stimDict"] = stimDict
#retValue = ITC.runITC(itcParm)
ITC.runITC([2, 4, 3, 0])
ITC.closeITC()
print "done"