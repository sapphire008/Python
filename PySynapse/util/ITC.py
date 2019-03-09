# revised 23 June 2015 BWS

from ctypes import *
import os.path as path
import time

class PyITC():
	def __init__(self, driverPath):
		self.initITC(driverPath)

	def initITC(self, driverPath):
		if path.isfile(driverPath):
			# typically driverPath is D:/LabWorld/Executables/ITCMM64.dll
			self.lib=cdll.LoadLibrary(driverPath)
			self.itc = self.lib.ITC18_GetStructureSize()
			self.itc = create_string_buffer(self.itc)
			status = self.lib.ITC18_Open(byref(self.itc), 0)
			if status != 0:
				print("problem with opening ITC18")
			status = self.lib.ITC18_Initialize(byref(self.itc), 0)
			if status != 0:
				print("problem with initialize ITC18")
			self.FIFOsize = self.lib.ITC18_GetFIFOSize(byref(self.itc))
			print("ITC started with FIFO = " + str(self.FIFOsize))
		else:
			print("could not find ITC18 dll at " + driverPath)
			self.lib = None

	def runITC(self, parmDict):
		# parmDict must contain values for these keys: msPerPoint, sweepWindowMs, extTrig0or1,
		# activeADCs (list of up to 8 numbers), activeDACs (list up to 4), ADCfullScale (list to match # of activeADCs),
		# ADCnames (list), ADCmultiplyFactors (list), stimDict (dict with TTL, DAC0 etc entries),
		# saveFileName (set to None for return trace Dict), protocol (Dict to convert to INI file)

		# ToDo add flag in parm dict to repeat last episode without reloading everything

		numInstructions = 4
		instructions = (c_int*numInstructions)()
		instructions[0] = 0x780 | 0x0
		instructions[1] = 0x780 | 0x800
		instructions[2] = 0x780 | 0x1000
		instructions[3] = 0x780 | 0x1800 | 0x4000 | 0x8000
		samplingRate = int((100. / numInstructions) / 1.25)
		ADCranges = (c_int*8)() # 0=10V 1=5V 2=2V 3=1V full scale
		waitForExtTrig = 0
		junkBeginning = 20 
		stimSize = 50 * numInstructions
		stimData = (c_short*stimSize)() # this needs to be a c_short to keep at 16 bit words
		dacValues = (c_short*numInstructions)()
		dacValues[0] = 0
		dacValues[1] = 0
		dacValues[2] = 2000
		dacValues[3] = 0
		for index in range(0, stimSize, numInstructions):
			for subindex in range(numInstructions):
				stimData[index + subindex] = dacValues[subindex]

		status = self.lib.ITC18_SetSequence(byref(self.itc), c_int(len(instructions)), byref(instructions))
		if status != 0:
			print("Problem with SetSequence ITC18 command")
		status = self.lib.ITC18_SetSamplingInterval(byref(self.itc), c_int(samplingRate))
		if status != 0:
			print("Problem with SetSamplingInterval ITC18 command")
		status = self.lib.ITC18_SetRange(byref(self.itc), byref(ADCranges))
		if status != 0:
			print("Problem with SetRange ITC18 command")
		status = self.lib.ITC18_InitializeAcquisition(byref(self.itc))
		if status != 0:
			print("Problem with InitializeAcquisition ITC18 command")
		status = self.lib.ITC18_WriteFIFO(byref(self.itc), c_int(stimSize), byref(stimData))
		if status != 0:
			print("Problem with WriteFIFO ITC18 command")
		status = self.lib.ITC18_Start(byref(self.itc), c_int(0), c_int(1), c_int(1), c_int(0))
		if status != 0:
			print("Problem with Start ITC18 command")
		time.sleep(1)
		status = self.lib.ITC18_Stop(byref(self.itc))
		if status != 0:
			print("Problem with Stop ITC18 command")


		

	def runITClowLevel(self, instructions, samplingRate, waitForExtTrig, ADCranges, stimData):
		status = self.lib.ITC18_SetSequence(byref(self.itc), c_int(len(instructions)), byref(instructions))
		if status != 0:
			print("Problem with SetSequence ITC18 command")
		status = self.lib.ITC18_SetSamplingInterval(byref(self.itc), c_int(samplingRate))
		if status != 0:
			print("Problem with SetSamplingInterval ITC18 command")
		status = self.lib.ITC18_SetRange(byref(self.itc), byref(ADCranges))
		if status != 0:
			print("Problem with SetRange ITC18 command")
		status = self.lib.ITC18_InitializeAcquisition(byref(self.itc))
		if status != 0:
			print("Problem with InitializeAcquisition ITC18 command")
		status = self.lib.ITC18_WriteFIFO(byref(self.itc), c_int(len(stimData)), byref(stimData))
		if status != 0:
			print("Problem with WriteFIFO ITC18 command")
		status = self.lib.ITC18_Start(byref(self.itc), c_int(0), c_int(1), c_int(1), c_int(0))
		if status != 0:
			print("Problem with Start ITC18 command")
		time.sleep(1)
		status = self.lib.ITC18_Stop(byref(self.itc))
		if status != 0:
			print("Problem with Stop ITC18 command")


	def setDACvalues(self, newDACvaluesList):
		numInstructions = 4
		instructions = (c_int*numInstructions)()
		instructions[0] = 1920
		instructions[1] = 1920 | 2048
		instructions[2] = 1920 | 4098
		instructions[3] = 1920 | 6144 | 16384 | 32768
		samplingRate = int((100. / numInstructions) / 1.25)
		ADCranges = (c_int*8)()
		waitForExtTrig = 0
		junkBeginning = 20 
		stimSize = 50 * numInstructions
		stimData = (c_int*stimSize)()
		for index in range(stimSize):
			stimData[index] = 2500
		retValue = self.runITClowLevel(instructions, samplingRate, waitForExtTrig, ADCranges, stimData)

	def closeITC(self):
		status = self.lib.ITC18_SetReadyLight(byref(self.itc), 0)
		if status != 0:
			print("Problem with turn off ready light")
		status = self.lib.ITC18_Close(byref(self.itc))
		if status != 0:
			print("Problem with ITC18 close")
		else:
			print("ITC18 closed")
            
if __name__ == '__main__':
    driverPath = 'D:/Edward/Documents/Assignments/Scripts/Python/PySynapse/resources/lib/ITCMM64.dll'
    ITC = PyITC(driverPath)
    
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
    print("done")