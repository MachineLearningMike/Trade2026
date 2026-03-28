from threading import Timer

from datetime import datetime, timedelta
import time as tm
import pytz

from config import *

class TimeMachine():

    nInstances = 0
    isSystemTimeSet = False

    tense = None
    originSec = None

    @classmethod
    def SetSystemTime( cls, originSec ):
        if cls.isSystemTimeSet is False:
            mili= round( (originSec - int(originSec)) * 1000 + 1 )
            gmtime = tm.gmtime( originSec )
            win32api.SetSystemTime(gmtime[0], gmtime[1], 0, gmtime[2], gmtime[3], gmtime[4], gmtime[5], mili) # Needs Administrator Permission
            cls.isSystemTimeSet = True
        return

    @classmethod
    def SetTense( cls, tense ):
        if cls.tense is None:
            cls.tense = tense
        elif cls.tense != tense:
            pass
        return

    @classmethod
    def SetOriginSec( cls, originSec ):
        if cls.originSec is None:
            cls.originSec = originSec
        elif cls.originSec != originSec:
            pass
        return
    
    #currentInterval = None

    def __init__(self, params):

        self.intervalSec = params['intervalSec']

        tense = params['tense']
        # snapSec = params['intervalSec'] * params['intervalsPerStep']

        if tense == Tense.Present:
            TimeMachine.SetTense(tense)
            TimeMachine.SetOriginSec(round(datetime.now().timestamp()))

        elif tense == Tense.Past or tense == Tense.Future:
            TimeMachine.SetTense(tense)
            TimeMachine.SetOriginSec(params['originSec'])

            self.currentIntervalForPastTense = 0

        TimeMachine.nInstances += 1

        return

    def __del__(self):
        TimeMachine.nInstances -= 1

    def TimeToIntervals(self, timeSec): # For any tense.
        intervalsFloat = (timeSec - TimeMachine.originSec) / self.intervalSec
        intervalsInt= round( intervalsFloat )
        fractionSec = (intervalsFloat - intervalsInt) * self.intervalSec
        return intervalsInt, fractionSec

    def GetCurrentInterval(self):
        if TimeMachine.tense is Tense.Present:
            intervalsInt, _ = self.TimeToIntervals(tm.time())
        else:
            intervalsInt = self.currentIntervalForPastTense # SleepForInterval() maintains this value.

        return intervalsInt

    def SleepForIntervals(self, callerInterval, intervals = 1): #------------------------------------- Performs a time machine.
        assert callerInterval is None or type(callerInterval) is int
        assert type(intervals) is int

        if TimeMachine.tense is Tense.Present:
            currentInterval, fractionSec = self.TimeToIntervals( tm.time() )
            if callerInterval is None: callerInterval = currentInterval
            deltaInterval = callerInterval + intervals - currentInterval
            if deltaInterval <= 0:
                if Config['Show_Traders']:
                    Print('============Time Machine: {} intervals lagging!'.format(-deltaInterval))
            #assert deltaInterval >= 0
            sleepSec = deltaInterval * self.intervalSec - fractionSec
            if sleepSec > 0:
                tm.sleep( sleepSec )
            else:
                if Config['Show_Traders']:
                    Print('---------- Time Machine: {} seconds lagging!'.format(-sleepSec))
            intervalsInt, _ = self.TimeToIntervals( tm.time() )
        else: # tense is Tense.Past or tense is Tense.Future
            if callerInterval is None: callerInterval = self.currentIntervalForPastTense
            self.currentIntervalForPastTense = callerInterval + intervals # GetCurrentInterval() uses this value.
            intervalsInt = self.currentIntervalForPastTense

        discretTimeSec = intervalsInt * self.intervalSec

        return intervalsInt, discretTimeSec

    def GetCurrentIntervalTime(self, callerInterval):
        if TimeMachine.tense is Tense.Present:
            ts = (TimeMachine.originSec + callerInterval * self.intervalSec)
        else:
            ts = (TimeMachine.originSec + callerInterval * self.intervalSec)

        return datetime.fromtimestamp(ts)
