# Business License, Mike

import threading
from datetime import datetime
import time as tm
import pytz

from abc import ABC, abstractmethod

from TimeMachine import *
from config import *

class ThreadOnTimeMachine(ABC):

    #debug = ""

    nInstances = 0
    instances = []

    def __init__(self, structuralParams, timingParams, steps = None):
        self._resume_flag = threading.Event() # The flag used to pause the thread
        self._run_flag = threading.Event() # Used to stop the thread
        self._contractorscanstart_flag = threading.Event()
        self._contractorsfinished_flag = threading.Event()
        self._icanstart_flag = threading.Event()
        self.thread = None

        self.timeMachine = TimeMachine(timingParams)
        self.currentInterval = 0
        self.intervalsPerStep = timingParams['intervalsPerStep']

        # Calls __setInstance__() of the immediate subclass, not that of sub-sub class.
        self.__setInstance__(structuralParams, timingParams) 

        self.steps = steps

        ThreadOnTimeMachine.nInstances += 1
        ThreadOnTimeMachine.instances.append(self)

        self.finished = False
        self.clients = []
        self.contractors = []

    def __del__(self):
        ThreadOnTimeMachine.nInstances -= 1

    def RegisterClient(self, client):
        if self.clients.index(client) < 0:
            self.clients.append(client)

    def UnregisterClient(self, client):
        if self.clients.index(client) >= 0:
            self.clients.remove(client)

    @abstractmethod
    def __setInstance__(self, structuralParams, timingParams):
        pass

    def Start(self): # Bottom-up start.

        if self.thread is not None: return

        if len(self.clients) <= 0:
            if Config['Show_Traders']:
                Print( "====== Bot starting a new round.\n" )
            self._icanstart_flag.set()
        else:
            self._icanstart_flag.clear()

        self._contractorsfinished_flag.clear()
        for contractor in self.contractors: contractor.Start()

        self._run_flag.set()
        self._resume_flag.set()

        self.thread = threading.Thread(target=self.ThreadFunciton, args=(self.steps,)) #--------------------------
        self.thread.start()

    def Stop(self): # Bottom-up stop.
        for contractor in self.contractors: contractor.Stop()
        #for contractor in self.contractors: contractor.Join()

        self._resume_flag.set() # Resume the thread from the suspended state, if it is alread suspended
        self._run_flag.clear() # Set to False
        #self.thread.join(). Parent is Joining, instead.
        #self.thread = None

        #if len(self.clients) <= 0:
        #    self.Join() # Self-join.
        #    Print('Root Timemachine joined()')

    def Join(self):
        for contractor in self.contractors: contractor.Join()
        self.thread.join()

    def Pause(self):
        self._resume_flag.clear() # Set to False to block the thread

    def Resume(self):
        self._resume_flag.set() # Set to True, let the thread stop blocking

    def OnContractorFinished(self, callingContractor):
        allfinished = True
        for contractor in self.contractors:
            if contractor._icanstart_flag.is_set():
                allfinished = False
                break
        if allfinished:
            self._contractorsfinished_flag.set()

        # last = 'All contractors done' if allfinished else 'A contractor done'
        #if len(callingContractor.contractors) > 0:
        # if Config['Show_Traders']:
        #     Print( "====== {} finished. {}\n".format( callingContractor.name, last) )

    def ThreadFunciton(self, steps):
        stepId = -1
        while steps is None or stepId < steps:
            stepId += 1
            self._resume_flag.wait()

            # self._contractorsfinished_flag.is_set() : No
            # self.__icanstart_flat.is_set() : No for all but the root.

            if self._run_flag.is_set():
                self._icanstart_flag.wait() # All is locked here except Root/Bot.

            if len(self.contractors) > 0:
                for contractor in self.contractors: contractor._icanstart_flag.set() # Free all the contractores, so they can finish.
                if self._run_flag.is_set():
                    self._contractorsfinished_flag.wait() # Wait until all contractores have finished
                    self._contractorsfinished_flag.clear() # Consume the fact that all contractors have finished.

            if self._run_flag.is_set():
                self.SingleStep(stepId) # As all contractores have finished, it's my turn to do.

            self._icanstart_flag.clear()

            if len(self.clients) > 0:
                for client in self.clients: client.OnContractorFinished(self)
                if not self._run_flag.is_set(): break
                if Config['Show_Traders']:
                    Print( "{}: --- {}: Finished {}-th round.".format( self.GetCurrentIntervalTime(), self.name, stepId ) )
                self.currentInterval, _ = self.timeMachine.SleepForIntervals(self.currentInterval, self.intervalsPerStep)
                if not self._run_flag.is_set(): break
            else:   # No clients, it's the Root/Bot.
                if Config['Show_Traders']:
                    Print( "{}: ============ Bot: Finished {}-th round. ============".format( self.GetCurrentIntervalTime(), stepId ) )
                self._icanstart_flag.set() # Only root can free itself up.

                if not self._run_flag.is_set(): break
                self.currentInterval, _ = self.timeMachine.SleepForIntervals(self.currentInterval, self.intervalsPerStep)
                if not self._run_flag.is_set(): break
                if Config['Show_Traders']:
                    Print( "====== Bot: Starting a new round. ---\n")

        if Config['Show_Traders']:
            Print('Time machine {} is quiting.'.format(self.name))
        #sys.exit(0) # Do NOT sys.exit(0), as join() is waiting.
        if len(self.contractors) > 0:
            for contractor in self.contractors: contractor._icanstart_flag.set() # Free all the contractores, so they can finish.

    def GetCurrentIntervalTime(self):
        return self.timeMachine.GetCurrentIntervalTime(self.currentInterval)

    @abstractmethod
    def SingleStep(self, stepId):
        # for n in range( 5 * pow(10, 6) ): a = pow(n, 0.5)
        pass

    # defined in Exchange/Utility.py
    def from_dt_local_to_dt_utc( dt_local ):
        now_timestamp = tm.time()
        offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)
        return (dt_local - offset).replace(tzinfo=pytz.utc)

    # defined in Exchange/Utility.py
    def from_dt_utc_to_dt_local( dt_utc ):
        now_timestamp = tm.time()
        offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)
        return (dt_utc + offset).replace(tzinfo=None)
