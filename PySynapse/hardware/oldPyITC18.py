from ctypes import *
from numpy import *
from pylab import *
from copy import *
import time
#from dummy import *

VC=0
IC=1
TRIGGER_LENGTH=80 #ms
COUNTSPERMV=2.**15/10240
FIFOSZ=3*(5*60*10000) #three times 300MB is three times 5 min
#FIFOSZ=(10*10000) #three times 300MB is three times 5 min
FIFOLAG=5000    #FIFOSZ NEEDS TO BE BIGGER THAN FIFOLAG
SAMPLINGRATE=10000
SLEEP=30
DEBUG_MODE=0
DT=(1./SAMPLINGRATE)*1000
NUMDACS=5
NUMADCS=5
EXTRA=50
DATASZ=2
REAL_MODE=5 #3 is for loopback, 5 is for real device
DEMO_MODE=3
REFRESHRATE=72
MICRONSPERPIXEL=1.54
LATER=0

CLOSED=0
OPEN=1
INITIALIZED=2
RUNNING=3
ERROR=4

class PyITC():
        def __init__(self,deviceType=5,deviceNumber=0,dbgDispDict=None):
                self.dbgDispDict=dbgDispDict
                self.printFlag=False
                #self.updateDbgDisp(beg=0,end=1000)
                self.fn='D:/LabWorld/Executables/ITCMM64.dll'
                self.lib=cdll.LoadLibrary(self.fn)
                self.WHERETYPE=POINTER(c_uint32)
                # self.lib.where.restype=self.WHERETYPE
                #self.lib=dummy()

                # self.recentNotify=0s
                self.parameters=dict()
                self.mode=[VC,VC]
                #(array([VC]),array([IC]))
                self.hold=(array([0,0,0,0,8],dtype=float32),array([0,0,0,0,8],dtype=float32))
                #outGain is a tuple: 0 is for VC and 1 is for IC
                #within each, the five element list is for the 4 DAC outputs and the 1 DO
                #self.outGain=(array([50., 50., 204.7/4, 204.7/4,1.]),array([2500., 2500., 204.7/4, 204.7/4,1.]))    #these gains used for LED stim on ch2-3
                self.outGain=(array([50., 50., 204.7/4, 1,1.]),array([2500., 2500., 204.7/4, 1,1.]))    # these gains used for one LED and one laser axis
                #inGain is a tuple: 0 is for VC and 1 is for IC
                #within each, the five element list is for E1raw, E1scaled, E2raw, E2scaled
                self.inGain=(array([0.1, 1./2500, 0.1, 1./2500]),array([1./500, 1./50, 1./500, 1./50]))

                self.setDeviceType(REAL_MODE)

                status=self.initAcq()
                if status!=0:
                        print 'Initialization error(%d)!' % status
                        self.stopAcq()
                        self.cleanupDLL()
                        return
                print 'Device initialized!'
                status=self.startAcq()
                #if status!=RUNNING:
                if status!=0:
                        print 'Start error(%d)!' % status
                        self.stopAcq()
                        self.cleanupDLL()
                        return
                print 'Acquisition started successfully!'
                self.setHolding()
                '''
                status=self.initAcq()
                i=0
                while status!=INITIALIZED and i<15:
                        status=self.initAcq()
                        i+=1
                if status!=INITIALIZED:
                        print 'Initialization error(%d)!' % status
                        return
                print 'Device initialized!'
                status=self.startAcq()
                if status!=RUNNING:
                        print 'Start error!'
                        return
                print 'Acquisition started successfully!'
                '''
        def myPrint(self,s,comma=''):
                if self.printFlag:
                        if comma=='':
                                print s
                        else:
                                print s,

        def setHolding(self,beg=0,n=None):
                if n==None:
                        n=FIFOSZ
                for i in range(4):
                        if i<2:
                                h=self.hold[self.mode[i]][i]*self.outGain[self.mode[i]][i]
                        else:
                                h=self.hold[self.mode[0]][i]*self.outGain[self.mode[0]][i]
                        tmp=uint16(self.mv2counts(h))*ones(n,dtype=uint16)
                        self.stickData(channel=i,pos=beg,n=n,ptr=tmp.ctypes.data_as(c_void_p))
                
                tmp=uint16(self.hold[0][4]*ones(n,dtype=uint16))
                self.stickData(channel=4,pos=beg,n=n,ptr=tmp.ctypes.data_as(c_void_p))
        def convertDataOut(self,do):
                for i in range(len(do)):
                        if i<2:
                                do[i]*=self.outGain[self.mode[i]][i]
                        else:
                                do[i]*=self.outGain[0][i]
        def convertDataIn(self,di):
                #print self.mode
                for i in range(len(di)-1):
                        di[i]*=self.inGain[self.mode[i/2]][i]
        def counts2mv(self,a):
                if a<2.**15:
                        a/=COUNTSPERMV
                else:
                        a=(a-2.**16)/COUNTSPERMV
                return a
        def counts2mvAll(self,di):
                for i in range(len(di)-1):
                        di[i]=float32(di[i])
                        di[i][di[i]<2.**15]/=COUNTSPERMV
                        di[i][di[i]>=2.**15]=(di[i][di[i]>=2.**15]-2.**16)/COUNTSPERMV
        def mv2counts(self,a):
                if a>0:
                        a*=COUNTSPERMV
                else:
                        a=(a+10240.)*COUNTSPERMV+2.**15
                return a
        def mv2countsAll(self,do):
                for i in range(len(do)):
                        if i<4:
                                do[i][do[i]>0]*=COUNTSPERMV
                                do[i][do[i]<=0]=(do[i][do[i]<=0]+10240)*COUNTSPERMV+2.**15
                        do[i]=uint16(do[i])
        def setDeviceType(self,type=DEMO_MODE):
                self.myPrint("Setting device type...",',')
                self.lib.setDeviceType(c_uint8(type))
                self.myPrint('success.')
        def initAcq(self,FIFOSZ=FIFOSZ,FIFOLAG=FIFOLAG,SAMPLINGRATE=SAMPLINGRATE,SLEEP=SLEEP,debugMode=DEBUG_MODE):
                self.myPrint('initAcq: ',',')
                tmp=int(self.lib.initAcq(c_uint32(FIFOSZ),c_uint32(FIFOLAG),c_uint32(SAMPLINGRATE),c_uint32(SLEEP),c_uint8(debugMode)))
                self.myPrint(tmp)
                return tmp
        def startAcq(self,timeout=7):
                self.myPrint('startAcq: ',',')
                self.lib.startAcq(c_int32(timeout))
                tmp=self.getStatus()
                self.myPrint(tmp)
                return tmp
        def getStatus(self):
                return int(self.lib.getStatus())
        def where(self):
                #whereType=c_uint32 * 10
                #tmp=whereType()
                tmp=self.lib.where()
                if tmp[0]==-1:
                        print "GOT NEGATIVE RESULT FROM WHERE()"
                        self.startAcq()
                        return self.where()
                w=tuple()
                for i in range(NUMDACS+NUMADCS):
                        w+=(tmp[i],)
                try:
                        #print type(w), w
                        w=array(w,dtype=int32)
                except:
                        print 'WHY???'
                        print sys.exc_info()
                        return None
                return w
        def stickData(self,channel=0,pos=0,n=FIFOSZ,flag='zero',ptr=None):
                if flag=='zero':
                        F=True
                else:
                        F=False
                if ptr!=None:
                        self.myPrint('Sticking data (CH%d POS: %d, N: %d, Z: %s...)' % (channel,pos,n,flag),',')
                        tmp=self.lib.stickData(c_uint8(channel),c_uint32(int32(pos)),c_uint32(n),c_int(F),ptr)
                        self.myPrint('done.')
                        if tmp<0:
                                self.startAcq()
                                return self.stickData(channel,pos,n,flag,ptr)
                        return tmp

        def grabData(self,channel=0,pos=0,n=FIFOSZ,flag='zero',ptr=None):
                if flag=='zero':
                        F=True
                else:
                        F=False
                if ptr!=None:
                        self.myPrint('Grabbing data (CH%d POS: %d, N: %d, Z: %s...)' % (channel,pos,n,flag),',')
                        tmp=self.lib.grabData(c_uint8(channel),c_uint32(int(pos)),c_uint32(n),c_int(F),ptr)
                        self.myPrint('done.')
                        if tmp<0:
                                self.startAcq()
                                return self.grabData(channel,pos,n,flag,ptr)
                        return tmp
        def getPointers(self):
                return self.lib.getPointers()
        def notify(self,n=0):
                self.recentNotify=n
                self.myPrint('Waiting for notify (%d)...' % n,',')
                tmp= self.lib.notify(n)
                self.myPrint('notified.')
                if tmp<0:
                        print 'Notify is waiting for acq to restart because of an error code.'
                        status=0
                        status=self.startAcq()
                        if status==0:
                                print 'Device restarted!'
                        else:
                                print 'Restart not possible! Exiting now.'
                                self.killAcq()
                                exit(1)
                        return None
                return tmp
        def notify_nowait(n=0):
                self.recentNotify=n
                tmp=self.lib.notify_nowait(n)
                if tmp<0:
                        self.startAcq()
                        return None
                return tmp
        def notify_poll(self):
                tmp=self.lib.notify_poll()
                if tmp>self.recentNotify:
                        self.startAcq()
                        return None
                return tmp
        def restartAcq(self):
                return int(self.lib.restartAcq())
        def stopAcq(self):
                return int(self.lib.stopAcq())
        def killAcq(self):
                return int(self.lib.killAcq())
        def cleanupDLL(self):
                return int(self.lib.cleanupdll())
        def thoroughTest(self):
                a=arange(start=0,stop=250000,dtype=uint16)
                while True:
                        for i in range(5):
                                b=ones(250000,dtype=uint16)
                                w=self.where()
                                pos=self.stickData(channel=i,pos=w[i]+25000,n=250000,ptr=a.ctypes.data_as(c_void_p),flag=True)
                                status=self.grabData(channel=i+5,pos=w[i]+25000,n=250000,ptr=b.ctypes.data_as(c_void_p))
                                if all(a==b):
                                        print 'Passed: %d' % pos
                                else:
                                        print 'Failed: %d' % pos
                                b=ones(250000,dtype=uint16)
                                pos=self.stickData(channel=i+5,pos=w[i+5]+25000,n=250000,ptr=a.ctypes.data_as(c_void_p),flag=True)
                                status=self.grabData(channel=i,pos=w[i+5]+25000,n=250000,ptr=b.ctypes.data_as(c_void_p))
                                if all(a==b):
                                        print 'Passed: %d' % pos
                                else:
                                        print 'Failed: %d' % pos
                        print
                        self.notify(100000)
        def outputTest(self):
                L=10*10e3
                #L=10
                LATER=20000
                #s=uint16(sin(arange(L)/25000.*2*pi)*7000.+8000)
                #s[0::2]+=8000
                #s[0::2]=0
                s=zeros(L,dtype=uint16)
                s[1::25000]=2000
                s[2::25000]=4000
                s[3::25000]=6000
                s[4::25000]=8000

                w=self.where()
                pos=self.stickData(channel=0,pos=w[0]+LATER,n=len(s),ptr=s.ctypes.data_as(c_void_p),flag='zero')
                start=time.time()
                self.notify(LATER+len(s))
                print 'ET: ',time.time()-start,'s.....',
                #self.grabData(channel=0,pos=w[0]+LATER+FIFOLAG,n=len(a),ptr=a.ctypes.data_as(c_void_p))
                #print 'A',a[:10]
        def stickAll(self,v):
                a=ones(FIFOSZ,dtype=uint16)*uint16(v)
                w=self.where()
                print 'Stick all...',
                pos=self.stickData(channel=0,pos=0,n=len(a),ptr=a.ctypes.data_as(c_void_p),flag='zero')
                print 'done.'
        def grabAll(self):
                a=zeros(FIFOSZ,dtype=uint16)
                print 'Grab all...',
                pos=self.grabData(channel=5,pos=0,n=len(a),ptr=a.ctypes.data_as(c_void_p),flag='zero')
                print a[:10],a[-10:]
                print 'done.'
