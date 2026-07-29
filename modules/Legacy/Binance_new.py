
import time as tm
from datetime import datetime, timedelta # Use datetime only, NOT date. Every datetime is local datetime if it has no 'utc' tag.
import pytz
from binance.client import Client
import math
import numpy as np
import pandas as pd
import json
import copy

import sys
import os

# Force Python to look in the Legacy directory FIRST for config
legacy_dir = os.path.dirname(os.path.abspath(__file__))
# Remove any existing config from sys.modules to force reload
if 'config' in sys.modules:
    del sys.modules['config']

# Add Legacy directory to the beginning of sys.path
sys.path.insert(0, legacy_dir)

# Now import from Legacy
# import config as legacy_config
from Utility import *

# Make Config and Print available
# Config = legacy_config.Config
# Print = legacy_config.Print

from ThreadOnTimeMachine import *

from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager as UnicornSockets
import logging

def Print(*argv, end="\n"):
    str = ''
    for arg in argv:
        str += arg
    lockPrint.acquire()
    print(str, end=end)
    lockPrint.release()

class ClientPool():

    def __init__(self, total):
        assert total >= 1
        self.total = total
        self.clients = []
        self.clientIdInUse = -1
        self.lock = threading.Lock()
        for n in range(total): self.clients.append( Client() )

    def Next(self):
        self.clientIdInUse = 0 if self.clientIdInUse >= len(self.clients) - 1 else self.clientIdInUse + 1
        return self.clients[self.clientIdInUse]

    def Repair(self, client):
        if client in self.clients:
            idx = self.clients.index(client)
        else: 
            idx = self.clientIdInUse
        self.clients[idx] = Client()
        return

class T_State():
    none = 0
    synced = 1
    syncing = 2
    live = 4
    no_missing = 8
    closed = 16
    changed = 32

class gstream():
    lock = None
    dynarray = None
    filepath = None
    stream_id = None
    tickbuffer = None
    t_state = None

    def __init__(self, _lock, _dynarray, _filepath, _stream_id, _tick_buffer, _t_state):
        lock = _lock
        self.dynarray = _dynarray
        self.filepath = _filepath
        self.stream_id = _stream_id
        self.tickbuffer = _tick_buffer
        self.t_state = _t_state

class Binance():

    _singleton = None
    _step_run = 0
    _running = False

    client = None   
    clientPool = None
    gstream_transceiver_thread = None
    gstream_demisser_thread = None

    stop_Maintain_GStreams = None
    stop_GStream_Transceiver_Thread_Function = None
    stop_GStream_Demisser_Thread_Function = None

    stream_lock = None
    gstreams = {}
    lock_gstream = None
    gstreams_ready = False
    gstream_datetime_utc = None # local time.
    gstream_lag_sec = 0
    unicorn_sockets_manager = None #UnicornSockets(exchange="binance.com") # took 4 weeks to remove '-futures', till July 14, 2021.
    streaming = False
    badkeys = []
    # tokenLookup = {}

    add_to_plot = None


    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instantiate(cls, connect=True):
        if cls._singleton is None:

            Binance.Initialize()
            cls.lock_gstream = threading.Lock()
           
            cls._singleton = cls.__new__(cls)
            
            if connect:
                client = Client(Config['api_key'], Config['api_secret']) #, tld = 'us', testnet = False)
                #client.API_URL = 'https://testnet.binance.vision/api'
                #client = Client()
                cls.client = client #, tld = 'us', testnet = False)

            travel_lag_mili = 150 # Specific to my location. Experimental.
            # This value is chosen so that the following evalustes to be the equal negative value.
            #self.GetExchangeTimeMili() - (datetime.utcnow().replace(tzinfo = pytz.utc)-datetime.utcfromtimestamp(0)).total_seconds() * 1000
            #(datetime.utcnow().replace(tzinfo = pytz.utc)-datetime.utcfromtimestamp(0)).total_seconds() * 1000 - self.GetExchangeTimeMili()

            # server_time = client.get_server_time()
            # milisec_since_epoch = server_time["serverTime"] + travel_lag_mili
            # gmtime=tm.gmtime(milisec_since_epoch/1000.0)
            # mili= round(milisec_since_epoch) % 1000
            # win32api.SetSystemTime(gmtime[0], gmtime[1], gmtime[6], gmtime[2], gmtime[3], gmtime[4], gmtime[5], mili) # Needed Administrator Permission

            if connect:
                Binance.clientPool = ClientPool(1)
                #Binance.clientPool.clients[0] = client        

        return cls._singleton


    @classmethod
    def Initialize(cls):
        Print("Initializing Binance singleton...")

        cls._singleton = None
        cls._step_run = 0
        cls._running = False

        cls.client = None   
        cls.clientPool = None
        cls.unicorn_sockets_manager = None
        cls.gstream_transceiver_thread = None
        cls.gstream_demisser_thread = None

        cls.stop_Maintain_GStreams = None
        cls.stop_GStream_Transceiver_Thread_Function = None
        cls.stop_GStream_Demisser_Thread_Function = None

        cls.stream_lock = threading.Lock()
        cls.gstreams = {}
        cls.lock_gstream = None
        cls.gstreams_ready = False
        cls.gstream_datetime_utc = None # local time.
        cls.gstream_lag_sec = 0
        #UnicornSockets(exchange="binance.com") # took 4 weeks to remove '-futures', till July 14.
        cls.unicorn_sockets_manager = None 
        cls.streaming = False

        cls.badkeys = []

        # cls.tokenLookup = readOrCreateTokenLookup()

        cls.demisser_gstream_heartbeat = None
        cls.transceiver_frame_heartbeat = None

        cls.add_to_plot = None

        return


    def Stop_Maintain_GStreams(self):
        signaled = None
        if Binance.stop_Maintain_GStreams is not None:
            Binance.stop_Maintain_GStreams = True
            signaled = True
        else:
            signaled = False

        return signaled


    def Maintain_GStreams(self):
        # The order in which the these two are called is a know-how.

        Binance._running = True
        self.GStream_Transceiver_Thread_Launcher(restore = False)

        if Config['Run_Demisser']:
            self.GStream_Demisser_Thread_Launcher(restore = False)

        Binance.stop_Maintain_GStreams = False

        def stop_transceiver_thread():
            Binance.stop_GStream_Transceiver_Thread_Function = True
            Binance.gstream_transceiver_thread.join()
            Print('gstream_transceiver_thread joined.')

        def stop_demisser_thread():
            Binance.stop_GStream_Demisser_Thread_Function = True
            Binance.gstream_demisser_thread.join()
            Print('gstream_demisser_thread joined.')


        try:
            while True:
                tm.sleep(5)

                if Binance.stop_Maintain_GStreams is None:
                    pass

                elif Binance.stop_Maintain_GStreams:
                    Binance._running = False

                    # Demisser, first.
                    if Config['Run_Demisser'] and Binance.stop_GStream_Demisser_Thread_Function is not None:
                        if Binance.gstream_demisser_thread.is_alive():
                            stop_demisser_thread()

                    if Binance.stop_GStream_Transceiver_Thread_Function is not None:
                        if Binance.gstream_transceiver_thread.is_alive():
                            stop_transceiver_thread()
                    break

                elif not Binance.stop_Maintain_GStreams:
                    # shall we initialize gstreams? =================================================

                    # Demisser, first.
                    if Config['Run_Demisser'] and Binance.demisser_gstream_heartbeat is not None \
                        and datetime.timestamp(datetime.now()) - Binance.demisser_gstream_heartbeat > 60:
                        # Tolerance = 300 seconds? Sometimes, 

                        Binance._running = False
                        if Binance.gstream_demisser_thread.is_alive():
                            stop_demisser_thread()
                        stop_GStream_Demisser_Thread_Function = False
                        self.GStream_Demisser_Thread_Launcher(restore = True)
                        Binance._running = True

                    if Binance.transceiver_frame_heartbeat is not None \
                        and datetime.timestamp(datetime.now()) - Binance.transceiver_frame_heartbeat > 60:
                        Binance._running = False
                        if Binance.gstream_transceiver_thread.is_alive():
                            stop_transceiver_thread()

                        stop_GStream_Transceiver_Thread_Function = False
                        self.GStream_Transceiver_Thread_Launcher(restore = True)
                        Binance._running = True

        except Exception as e:
            Print('Maintain_GStreams() with exception: ')
            Binance.stop_Maintain_GStreams = None
            sys.exit(1) # This thread will be relaunched later.


        # Do NOT sys.exit(0), as join() is waiting.
        Print('Maintain_GStreams() stopped.')
        Binance.stop_Maintain_GStreams = None
        return


    def GStream_Transceiver_Thread_Launcher(self, restore = False):
        # Re-create sockets_manager each time?
        Binance.unicorn_sockets_manager = UnicornSockets(exchange = Config['exchange']) # It took 4 weeks to remove '-futures', till July 14, 2021.
        
        Binance.gstream_transceiver_thread = threading.Thread(target = self.GStream_Transceiver_Thread_Function, args=())
        Binance.gstream_transceiver_thread.start()

        # for key, gs in Binance.gstreams.items():
        #     (dataType, symbol, interval) = key.split('.')
        #     self.Maintain_Existence_Of_GStream(dataType, symbol, interval, live = bool(gs.t_state & T_State.live), max = 200)

        self.Dynamize_Static_GStreams_v2( live = True, restore = restore )
        self.Dynamize_Static_GStreams_v2( live = False, restore = restore )

        return


    def GStream_Transceiver_Thread_Function(self):
        # Design requirements: Make it light. Don't let it halt on io.
        sleepSec = 0.001

        Binance.stop_GStream_Transceiver_Thread_Function = False

        try:

            while Config['Stream_Price']:
                Binance.transceiver_frame_heartbeat = datetime.timestamp(datetime.now())

                nSleepSec = 0

                if Binance.stop_GStream_Transceiver_Thread_Function is None:
                    pass
                elif Binance.stop_GStream_Transceiver_Thread_Function:
                    Binance.unicorn_sockets_manager.stop_manager_with_all_streams()
                    break
                elif Binance.unicorn_sockets_manager.is_manager_stopping(): # hasattr(Binance, "unicorn_sockets_manager") and 
                    break  # sys.exit(0) Do NOT exit. Join() is waiting for this normal quiting.
                elif not Binance.stop_GStream_Transceiver_Thread_Function:
                    frame = Binance.unicorn_sockets_manager.pop_stream_data_from_stream_buffer()
                    if frame is False:
                        time.sleep(sleepSec) #============= Expensive OS switching.!!!!!!!!!! so frequent.
                        nSleepSec += sleepSec
                    else:
                        # data_consumed = self.Dispatch_Streaming_DataPoint(frame)

                    #def Dispatch_Streaming_DataPoint(self, frame):
                        # Design requirements: Make it light. Don't let it halt on io.
                        # https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data
                        # https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#klinecandlestick-streams
                        
                        if type(frame) == str: frame = json.loads(frame)
                        assert type(frame) == dict

                        data_consumed = False
                        stream_str = frame.get('stream', None)
                        if stream_str is not None:
                            self.streaming = True
                            dataType, symbol, interval = Binance.Unicorn_dataType_Symbol_Interval(stream_str)
                            key = dataType + '.' + symbol + '.' + interval

                            if Binance.gstreams.get(key, None) is None:
                                # gstreams are unsubscribed before removed from Binance.gstreams.
                                # Some possible dust frames may cause this.
                                self.Maintain_Existence_Of_GStream(dataType, symbol, interval, live = True, max = 200)
                            else:
                                data_consumed = True
                                frame = frame['data']

                                event_mili = frame['E']
                                candle_open_mili = frame['k']['t']

                                # if Binance.gstream_datetime_utc is None or now_closed:
                                Binance.gstream_datetime_utc = datetime.fromtimestamp(candle_open_mili/1000).replace(tzinfo=pytz.utc)
                                utcnow = datetime.utcnow()               
                                Binance.gstream_lag_sec = (utcnow - datetime.utcfromtimestamp(event_mili/1000)).total_seconds()
                                    
                                gs = Binance.gstreams[key]

                                live = bool(gs.t_state & T_State.live)
                                prev_closed = bool(gs.t_state & T_State.closed) # what if ...

                                report = 'success'

                                if dataType == 'klines':  # The Kline/Candlestick Stream push updates to the current klines/candlestick every second.
                                    assert symbol.upper() == frame['s']

                                    kstream = frame['k']

                                    candle_open_mili = kstream['t']
                                    now_closed = bool(kstream['x'])

                                    # new_row_raw = [candle_open_mili, open, high, low, close, volume, closetime_stream, vquote, ntrades, vbasetakerbuy, vquotetakerbuy, ignore]
                                    new_row = [kstream['t'], kstream['o'], kstream['h'], kstream['l'], kstream['c'], kstream['v'], 
                                    kstream['T'], kstream['q'], kstream['n'], kstream['V'], kstream['Q'], kstream['B']]
                                    [new_row] = parse_datalines_to_datalines(dataType, [new_row]) # Convert string to float.

                                    intervalmili = intervalToMilliseconds(interval)

                                    # with gs.lock: # Does not work.

                                    nExisting_rows = gs.dynarray.shape[0] # dynarray.shape[0]
                                    overriden = None
                                    misses = None

                                    #------------------------------------------------------------------------------
                                    # Here, we achieve and make sure that:
                                    # Binance-streamed price data is stored/cached in time order, although
                                    #  - there can be missing data points, and
                                    #  - Binance streams price data in reverse time order, VERY rarely.
                                    # Demissing functions will rely on this assumption, to save compute resources.
                                    #-------------------------------------------------------------------------------

                                    # Missing rows will be handled somewhere else.
                                    if live:
                                        if prev_closed:  # now_closed or not. dynarray empty or not.
                                            if nExisting_rows > 0:
                                                overriden = False
                                                misses = round( (new_row[0] - gs.dynarray[-1, 0]) / intervalmili ) - 1
                                                if misses >= 0: # Demisser will be free from non-increasing frames.
                                                    gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)   
                                                    gs.t_state = gs.t_state | T_State.changed
                                            else:
                                                gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                gs.t_state = gs.t_state | T_State.changed

                                        else:   # override it, now_closed or not.
                                            if nExisting_rows > 0:
                                                overriden = True
                                                misses = round( (new_row[0] - gs.dynarray[-1, 0]) / intervalmili )
                                                if misses >= 0: # Deimsser will be free from non-increasing frames.
                                                    gs.dynarray[-1] = new_row
                                                    gs.t_state = gs.t_state | T_State.changed
                                            else:   # this is impossible, as empty dynarrays are prev_closed by default.
                                                gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                gs.t_state = gs.t_state | T_State.changed
                                    else:
                                        if prev_closed:
                                            if now_closed:
                                                if nExisting_rows > 0:
                                                    if round( (new_row[0] - gs.dynarray[-1, 0]) / intervalmili ) >= 1:
                                                        gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                        gs.t_state = gs.t_state | T_State.changed
                                                else:
                                                    gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                    gs.t_state = gs.t_state | T_State.changed
                                            else:
                                                pass    # ignore, so prev_closed is impossible.
                                        else:   # although this should not happen, we respect now_closed.
                                            if now_closed:
                                                if nExisting_rows > 0:
                                                    gs.dynarray[-1] = new_row
                                                else:   # this is impossible, as empty dynarrays are prev_closed by default.
                                                    gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                gs.t_state = gs.t_state | T_State.changed
                                            else:
                                                pass    # ignore, so prev_closed is impossible.

                                    gs.t_state =  (gs.t_state & ~T_State.closed) | T_State.closed if now_closed else  gs.t_state & ~T_State.closed

                                    #=============================================================================================================
                                    now_closed = bool(gs.t_state & T_State.closed)

                                    if  Config['Show_Transceiver']:
                                        Print(
                                            frame['s'].ljust(10),
                                            frame['k']['i'].zfill(3),
                                            ', C' if now_closed else ',  ',
                                            ', N ' + utcnow.strftime('%M:%S'),
                                            ', R {:.0f}m'.format((utcnow.replace(tzinfo=pytz.utc)-Config['start_time_utc']).total_seconds()/60),
                                            ', L {}s'.format(str(round(Binance.gstream_lag_sec))),
                                            ', S ' + datetime.utcfromtimestamp(candle_open_mili/1000).strftime('%M:%S'),
                                            ', E ' + datetime.utcfromtimestamp(event_mili/1000).strftime('%M:%S'),
                                            ', {}'.format('Over' if overriden else 'Next'),
                                            ', M {}'.format(misses),
                                        )
                                    #=============================================================================================================

                                if Binance.add_to_plot is not None:
                                    Binance.add_to_plot(symbol, candle_open_mili, new_row[4] )
                                
                                #TODO. Temporarilly commented out.
                                #if launching_frame_delagging_required: #=====================================================================
                                #    self.Launch_Max_Grow_Static_GStream(key) # This will take care of t_state

                        else: # this is a control signal.
                            self.streaming = False

                        
                        #return data_consumed

                        time.sleep(0.01) # Yield chance to others.
                        if not data_consumed:
                            # logging
                            #Binance.unicorn_sockets_manager.delete_stream_from_stream_list(stream_id) # How to find stream_id?
                            pass

                if nSleepSec <= 0: tm.sleep(sleepSec) # This requires OS thread switching, which is resource-exspensive. Try await.

            # Do NOT sys.exit(), as join() is waiting.
            return

        except Exception as e:
            Binance.gstream_transceiver_thread._is_stopped = True
            Binance.unicorn_sockets_manager.stop_manager_with_all_streams()
            sys.exit(1) # This thread will be relaunched later.


    def GStream_Demisser_Thread_Launcher(self, restore = False):
        Binance.gstream_demisser_thread = threading.Thread(target=self.GStream_Demisser_Thread_Function, args=())
        Binance.gstream_demisser_thread.start()

    def DemissGStream(self, key, nKeys, nMissings):
        """
        The transceiver function claims as follows:

        #------------------------------------------------------------------------------
        # Here, we achieve and make sure that:
        # Binance-streamed price data is stored/cached in time order, although
        #  - there can be missing data points, and
        #  - Binance streams price data in reverse time order, VERY rarely.
        # Demissing functions will rely on this assumption, to save compute resources.
        #-------------------------------------------------------------------------------
        For each existing gstream:
        Goal 1: Make sure to prepend the gstream with currently yesterday data, on a time-ordered and non-missing basis.
        Goal 2: Fill in missing data intervals of transceived data, with time-ordered non-missing price data.

        We must assume competition for access to gstreams' dynarray (numpy array) 
        between this Dismisser, Transceiver, and trading indicatore.
        treadhing.Lock does not work, and we have to find a workaround to avoid conflict.
        """
        successful = True
        nKeys1 = nKeys; nMissings1 = nMissings; next = None

        gs = Binance.gstreams[key]

        if key in Binance.badkeys:
            return nKeys, nMissings, "continue" # violating coding rules.       
        nKeys += 1

        (dataType, symbol, interval) = key.split('.')
        monitor = symbol.ljust(12)
        Binance.demisser_gstream_heartbeat = datetime.timestamp(datetime.now())

        slot = get_timeslot(dataType)
        interval_mili = intervalToMilliseconds(interval)

        binance_wait_sec = 0 if nKeys < 180 else 1 if nKeys < 200 else Config['binance_wait_sec'] # coming from experience.

        no_missing = True

        live = bool(gs.t_state & T_State.live)
        closed = bool(gs.t_state & T_State.closed) # what if ...

        # This should the only place to pick gs.dynarray.shape[0],
        # and all the following lines should stick to this value.
        # This is to avoid conflicts coming from competetion for gstream access  
        # between Transceiver, Dimisser, and indicators.
        # The competition aroses because threading.Lock (gstream.lock) does not work
        # for unknown reasons. 
        initially_existing_rows = gs.dynarray.shape[0]
        search_start = 0
        search_end = initially_existing_rows - 1 if live and not closed else initially_existing_rows

        yesterday_first_ts_mili = round(datetime.timestamp(get_previoius_day_start(datetime.utcnow().replace(tzinfo=pytz.utc)))*1000)
        # We have to collect yesterday's price history in to gstream memory, at the expense of memory drain,
        # because yesterday's Binance file is not available until 9 o'clock in the morning.

        if initially_existing_rows > 0:  # The dataframe has at least one row.
            first_ts_mili_excl= round(gs.dynarray[0][slot])
        else: # has no rows.
            # continue    # Better.
            first_ts_mili_excl = round(datetime.timestamp(datetime.utcnow())*1000/interval_mili) * interval_mili + interval_mili # probably the current candle
            prepended_to_now = True

        nRowsToPrepend = 0
        if yesterday_first_ts_mili + interval_mili < first_ts_mili_excl: # We need to prepend the dynarray with stuff.
            nRowsToPrepend = round((first_ts_mili_excl - yesterday_first_ts_mili) / interval_mili) # first_ts_mili_excl is exclusive
            if nRowsToPrepend > 0:
                rows = exact_fetch(self.clientPool, binance_wait_sec, dataType, symbol, interval, yesterday_first_ts_mili, first_ts_mili_excl - interval_mili , nRowsToPrepend, interval_mili)
                # with gs.lock: # Does not work.
                if isinstance(rows, list):  # this filters out rows being None.
                    nRows = len(rows)   # Something like "str object has not length" error message happens if rows == None, taking a full day of debugging.
                    if nRows != nRowsToPrepend:
                        if Config['Show_Demisser']:
                            Print("??? Fetching failed. {} blacklisted.".format(key))
                        Binance.badkeys.append(key)
                        return nKeys, nMissings, "continue" # I know it's violationg the coding rule.

                    try:
                        rows = parse_datalines_to_datalines(dataType, rows) # Convert string to float.
                        array = np.array(rows) # dtype should be float64.
                        array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
                        assert is_one_incremental( array[:, slot].astype(np.int64) / interval_mili ), \
                            "The rows fetched for prepending is not one-incremental. {}".format(key)    # time-ordered no-missing.

                        gs.dynarray = np.insert(gs.dynarray, 0, array, axis=0)
                        monitor += ", pre {:>4}".format(nRowsToPrepend)
                        search_start = nRowsToPrepend - 1     # update search_start.
                        # No! closed = True   # as we're sure the fetched rows are all closed candles. 

                    except Exception as e:
                        Print("~~~~~~~~~~~~~~~~~~~~~~~ Exception in DemissGStream core: \n", str(e))
                else:
                    no_missing = False  # Try it at later rounds.
                    monitor += ",".ljust(10)    # in place of ", pre {:>4}".format(nRowsToPrepend)
            else:
                Print("~~~~~~~~~~~~~~~~~~~~~~~ Exception in DemissGStream: \n")

        elif first_ts_mili_excl < yesterday_first_ts_mili: # Remove early rows from dynarray, to save memory.
            gs.dynarray = gs.dynarray[ gs.dynarray[:, slot] >= yesterday_first_ts_mili - 1 ]    # not tested.
            monitor += ",".ljust(10)    # in place of ", pre {:>4}".format(nRowsToPrepend)

        #----------------- Fill in the missing, intermediate rows.
        if no_missing: # Not tested.
            # with gs.lock: # Does not work.
            nRows = initially_existing_rows + nRowsToPrepend # gs.dynarray.shape[0]
            search_end = nRows - 1 if live and not closed else nRows
            assert search_end >= 1, "search_end < 1."

            # if not is_one_incremental( gs.dynarray[search_start:search_end, slot].astype(np.int64) / interval_mili ):
            #     Print("Demiss: One_Incremental rule violated. Correcting...")
            #     gs.dynarray = snap_sort_deduplicate_in_opening_time(gs.dynarray[search_start:search_end, slot], interval_mili)   # This is a cross-file check.

            # start_mili= round(gs.dynarray[search_start,0]*1000/interval_mili) * interval_mili
            # end_mili= round(gs.dynarray[search_end-1,0]*1000/interval_mili) * interval_mili
            # nRowsExpected= round((end_mili - start_mili) / interval_mili) + 1

            # if search_end - search_start < nRowsExpected:
            #     Print("Demiss: Final: Missing points found. Correcting...")
            #     gs.dynarray, nForged, successful = create_missing_prices_inland(self.clientPool, Config['binance_wait_sec'], gs.dynarray, dataType, symbol, interval, start_mili, end_mili + interval_mili)
            #     successful = gs.dynarray.shape[0] == nRowsExpected

            if (round(gs.dynarray[search_end-1, slot]) - round(gs.dynarray[search_start, slot])) == (search_end - 1 - search_start) * interval_mili:
                assert is_one_incremental( gs.dynarray[search_start:search_end, slot].astype(np.int64) / interval_mili ), \
                    "The rows following prepending is not one-incremental."    # time-ordered no-missing.           
            else: # very rare but very expensive
                # Acheives the Goal 2. #######################################
                gs.dynarray, nForged, no_missing \
                = create_missing_prices_inland(Binance.clientPool, binance_wait_sec, gs.dynarray, dataType, symbol, interval, 0, 0, ignore_last = live and closed)
                if nForged > 0:
                    monitor += ", F {}".format(nForged)

        if no_missing:
            # Mask out these assertions later: it's logically proven and expensive to compute, although it a complete proof.
            # ----- This may not hold sometimes for unknown reasons.
            # assert round(gs.dynarray[search_end-1, slot]) - round(gs.dynarray[0, slot]) == (search_end-1) * interval_mili, \
            #     "Reported no missing, but the time-distance between the 1st and the last closed candles is wrong."
            try:
                assert is_one_incremental( gs.dynarray[0:search_end, slot].astype(np.int64) / interval_mili ), \
                    "{} candles are not one-incremental. Correcting...".format(symbol)    # time-ordered no-missing.
            except:
                pass
        else:
            gs.t_state = gs.t_state | T_State.changed

        gs = Binance.gstreams[key] # because keys are dynamically added/removed. Existence?
        if no_missing != bool(gs.t_state & T_State.no_missing): # missing state changed
            if no_missing: # good change
                gs.t_state = gs.t_state | T_State.no_missing
                monitor += ", No-missing".ljust(12) #", <Missing solved>"
            else:   # bad change
                gs.t_state = gs.t_state & ~T_State.no_missing
                monitor += ", Missing".ljust(12) #", <Missing found>"
                nMissings += 1
        elif bool(gs.t_state & T_State.no_missing):
            monitor += ", No-missing".ljust(12) #", <No-missing again>"
        else:
            monitor += ", Missing".ljust(12) #", <Missing again>"
    
        monitor += ", nKey {:>3}, N {:>4}".format(nKeys, gs.dynarray.shape[0])
        monitor = "--- Demiss: " + monitor

        if Config['Show_Demisser']:
            Print(monitor) ############################# Conclusion of the demissing work on this key.

        #============= This is VERY important design decision to yield time resources to Transceiver thread. Transceiver would increasingly lag behind the stream.
        if Binance.stop_GStream_Demisser_Thread_Function: 
            return nKeys, nMissings, "break" # For responsiveness.

        items = Binance.gstreams.items()
        nGStreams = len(items) if items is not None else 0
        tm.sleep(
            Config['Bias_Sec_per_Demiss'] / nGStreams
            + Config['Linear_Sec_per_Lag_X_#gstream'] * Binance.gstream_lag_sec * nGStreams
        )

        if Binance.gstream_datetime_utc is None:    # in case GStream_Transceiver_Thread_Function failed to capture it.
            Binance.gstream_datetime_utc = from_dt_local_to_dt_utc( datetime.now() )

        return nKeys, nMissings, "next"


    def GStream_Demisser_Thread_Function(self):
        # Design requirements: Make it light. Don't let it halt on io.
        Binance.stop_GStream_Demisser_Thread_Function = False

        try:
            while Config['Run_Demisser']:
                if Config['Show_Demisser']:
                    Print('******* new demisser cycle --------\n')

                nKeys = 0
                nMissings = 0

                if Binance.stop_GStream_Demisser_Thread_Function is None:
                    pass
                elif Binance.stop_GStream_Demisser_Thread_Function:                    
                    break
                elif not Binance.stop_GStream_Demisser_Thread_Function:
                    # items = Binance.gstreams.items()
                    gstreams = Binance.gstreams.copy()

                    for key, gs in gstreams.items():
                        nKeys, nMissings, next = self.DemissGStream(key, nKeys, nMissings)

                        if next == "break": break
                        if next == "continue": continue
                        if next == "next": pass

                    # end of "for key, gs in ..."

                # end of demissing cycle.

                if not Binance.stop_GStream_Demisser_Thread_Function:

                    if Config['Show_Demisser']:
                        Print('############# Demisser cycle done: missing {} of {} keys, with {} bad key(s)\n'.format(nMissings, nKeys, len(Binance.badkeys)))
                    if nMissings <= 0:
                        if Config['Show_Demisser']:
                            Print('------------GStreams Ready --------\n')
                        Binance.gstreams_ready = True
                    else:
                        Binance.gstreams_ready = False

            # end of "while ...""

        # end of "try..."
        except Exception as e:
            Binance.gstream_demisser_thread._is_stopped = True
            Print("~~~~~~~~~~~~~~~~~~~~~~~ Exception in GStream_Demisser_Thread_Function: \n", str(e))

        return


    def Dynamize_Static_GStreams_v2(self, live = True, restore = False):

        # with Binance.lock_gstream:
        Binance.gstreams = { key: value for (key, value) in sorted( Binance.gstreams.items(), key=lambda x: x[1].t_state & T_State.live, reverse=True) }
 
        slot_groups =[]
        # with Binance.lock_gstream: # Fast enough is this block.
        for key, gs in Binance.gstreams.items():
            # if ( bool(gs.t_state & T_State.live) == live ) and ( not ( restore or gs.t_state & (T_State.synced | T_State.syncing) ) or restore ) :
            if ( bool(gs.t_state & T_State.live) == live ) and ( restore or not gs.t_state & (T_State.synced | T_State.syncing) ) :

                [dataType, symbol, interval] = key.split('.')
                market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)

                group_shared = False
                for slot_group in slot_groups:
                    channels = slot_group['channels']
                    markets = slot_group['markets']
                    if len(channels) * len(markets) < Config['max_subscriptions_per_stream']:
                        if len(channels) == 1 and channels[0] == channel:
                            markets.append(market)
                            group_shared = True
                        elif len(markets) == 1 and markets[0] == market:
                            channels.append[channel]
                            group_shared = True
                    if group_shared: break
                    
                if not group_shared:
                    slot_group = {'channels': [channel], 'markets' : [market] }
                    slot_groups.append(slot_group)

        for slot_group in slot_groups:
            tm.sleep(Config['binance_wait_sec'])
            stream_id = self.unicorn_sockets_manager.create_stream( slot_group['channels'], slot_group['markets'] )
            if Config['Show_Demisser']:
                Print("---{} gstreams on a new price stream".format( len(slot_group['channels']) * len(slot_group['markets']) ))
            if stream_id is not None:
                # with Binance.lock_gstream:
                for channel in slot_group['channels']:
                    for market in slot_group['markets']:
                        dataType, symbol, interval = Binance.Unicorn_dataType_Symbol_Interval_From_Channel_Market(channel, market)
                        key = dataType + '.' + symbol + '.' + interval
                        if Binance.gstreams.get(key, None) is not None:
                            Binance.gstreams[key].stream_id = stream_id
            else:
                raise Exception('Failed in an attempt to create stream.')
        return


    @classmethod
    def Start_Add_To_Plot(cls, add_to_plot):
        cls.add_to_plot = add_to_plot

    def GetExchangeTimeMili(cls):
        super().GetExchangeTimeMili()
        server_time = cls.client.get_server_time() # Miliseconds since epoch.
        miliseconds_from_epoch= round(server_time["serverTime"])

        return miliseconds_from_epoch


    def Maintain_Existence_Of_GStream(self, dataType, symbol, interval, live = False, max = 200):
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        key = dataType + '.' + symbol + '.' + interval

        # Make sure it exists on Binance.gstreams.
        gs = None
        if Binance.gstreams.get(key, None) is not None:
            gs = Binance.gstreams[key]
            gs.t_state = gs.t_state | T_State.live if live else gs.t_state & ~T_State.none
            # if Config['Show_Transceiver'] or Config['Show_Traders']: 
            # Print("+++++ {} already exists in gstreams.".format(key))
        elif len(Binance.gstreams.items()) < max:
            gs = self.Create_Static_GStream_v2(dataType, symbol, interval, live=live)
            Binance.gstreams[key] = gs
            if Config['Show_Transceiver'] or Config['Show_Traders']: 
                Print("+++++ {} is newly added to gstreams.".format(key))

            # Activate it if needed.
            if Binance.unicorn_sockets_manager is not None:
                if Binance._running and gs is not None:    # Activate it. unicorn_stream_manager is only avaiable when Binance._running.
                    # # with Binance.lock_gstream:
                    # if self.Is_On_Active_Streams(dataType, symbol, interval):
                    #     # if Config['Show_Transceiver'] or Config['Show_Traders']: 
                    #     # Print("+++++ {} exists in price streams. Removing redundants...".format(key))
                    #     self.Remove_Duplicates_On_Active_Streams(dataType, symbol, interval)
                    #     assert Binance.gstreams.get(key, None) is not None
                    # else:

                    if Config['Show_Transceiver'] or Config['Show_Traders']: 
                        Print("+++++ {} is being added to Binance streams...".format(key))
                    self.Subscribe_to_Existing_or_New_Active_Stream(key)

        return key

    def Create_Static_GStream_v2(self, dataType, symbol, interval, live=False):
        dynarray = create_empty_dynarray()
        # 12: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data
        filepath, stream_id = None, None
        tickbuffer, t_state = [], T_State.none | (T_State.live if live else T_State.none) | T_State.closed | T_State.no_missing

        return gstream(threading.Lock(), dynarray, filepath, stream_id, tickbuffer, t_state) #stream_str (eg, 'ethusdt@kline_1m')

    def Unsubscribe_GStream(self, dataType, symbol, interval):
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        unsubscribed = False

        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)

        with Binance.stream_lock:
            stream_ids = [id for (id, _) in Binance.unicorn_sockets_manager.get_stream_list().items()]
            for stream_id in stream_ids:
                stream_info = Binance.unicorn_sockets_manager.get_stream_info(stream_id)
                if stream_info['channels'] is not None and len(stream_info['channels']) == 1 and stream_info['channels'][0] == channel :
                    unsubscribed = Binance.unicorn_sockets_manager.unsubscribe_from_stream(stream_id, [], [market])
                elif stream_info['markets'] is not None and len(stream_info['market']) == 1 and stream_info['markets'][0] == market :
                    unsubscribed = Binance.unicorn_sockets_manager.unsubscribe_from_stream(stream_id, [channel], [])
        return


    def Is_On_Active_Streams(self, dataType, symbol, interval):
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        unsubscribed = False

        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)

        found_active = False
        stream_list = Binance.unicorn_sockets_manager.get_stream_list()
        if stream_list is not None and isinstance(stream_list, dict):
            with Binance.stream_lock:
                stream_ids = [id for (id, _) in Binance.unicorn_sockets_manager.get_stream_list().items()]
                for stream_id in stream_ids:
                    stream_info = Binance.unicorn_sockets_manager.get_stream_info(stream_id)
                    if stream_info['channels'] is not None and stream_info['markets'] is not None:
                        if channel in stream_info['channels'] and market in stream_info['markets']:
                            found_active = True
                            break

        return found_active


    def Remove_Duplicates_On_Active_Streams(self, dataType, symbol, interval):
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        unsubscribed = False

        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)

        found_active = False
        with Binance.stream_lock:
            stream_ids = [id for (id, _) in Binance.unicorn_sockets_manager.get_stream_list().items()]
            for stream_id in stream_ids:
                stream_info = Binance.unicorn_sockets_manager.get_stream_info(stream_id)
                if channel in stream_info['channels'] and market in stream_info['markets']:
                    if not found_active:
                        found_active = True
                    else:
                        if stream_info['channels'] is not None and len(stream_info['channels']) == 1 and stream_info['channels'][0] == channel :
                            unsubscribed = Binance.unicorn_sockets_manager.unsubscribe_from_stream(stream_id, [], [market])
                        elif stream_info['markets'] is not None and len(stream_info['market']) == 1 and stream_info['markets'][0] == market :
                            unsubscribed = Binance.unicorn_sockets_manager.unsubscribe_from_stream(stream_id, [channel], [])
        
        return


    def Remove_GStream(self, dataType, symbol, interval):
        Print("-----------Remove_GStream")

        symbol = str.upper(symbol)
        interval = str.lower(interval)
        unsubscribed = False

        self.Unsubscribe_GStream(dataType, symbol, interval)

        key = dataType + '.' + symbol + '.' + interval
        if Binance.gstreams.get(key, None) is not None:
            del Binance.gstreams[key]

        return
    

    def Download_Monthly_Data(self, dataTypes, symbols, intervals, start, end):
        """
        start:  local datetime
        end:    local datetime, or None representing now.
        """
        
        start_utc = from_dt_local_to_dt_utc(start)
        if end is None: end = datetime.now()
        end_utc = from_dt_local_to_dt_utc(end)

        start_utc = datetime( start_utc.year, start_utc.month, 1 ) - timedelta(days=1)
        end_utc = datetime( end_utc.year, end_utc.month + 1, 1 )

        paths = {}
        symbolcounter = 0
        for dataType in dataTypes:
            for symbol in symbols:
                Print("[{}/{}] - start download monthly {} {}.".format(symbolcounter+1, len(symbols), symbol, dataType))
                for interval in intervals:
                    paths_list = []
                    for year in range( start_utc.year, end_utc.year + 1):
                        for month in range(1, 13):
                            current_utc = datetime(year, month, 2) # 2 not 1.
                            if start_utc <= current_utc and current_utc <= end_utc:
                                path = download_file(current_utc, dataType, symbol, interval)
                                paths_list.append(path)
                    paths[dataType+'.'+symbol+'.'+interval] = paths_list       
                symbolcounter += 1

        return paths


    def Download_Daily_Data(self, dataTypes, symbols, intervals, start, end):

        start_utc = from_dt_local_to_dt_utc(start)
        if end is None: end = datetime.now()
        end_utc = from_dt_local_to_dt_utc(end)
       
        start_utc = datetime( start_utc.year, start_utc.month, start_utc.day ) - timedelta(hours=1)
        end_utc = datetime( end_utc.year, end_utc.month, end_utc.day ) + timedelta(hours=1)

        paths = {}
        symbolcounter = 0
        for dataType in dataTypes:
            for symbol in symbols:
                Print("[{}/{}] - start download daily {} {}.".format(symbolcounter+1, len(symbols), symbol, dataType))
                for interval in intervals:
                    paths_list = []
                    for days in range( (end_utc-start_utc).days + 2 ):  
                        current_utc = datetime(start_utc.year, start_utc.month, start_utc.day) + timedelta(days=days)
                        if start_utc <= current_utc and current_utc <= end_utc:
                            path = download_file(current_utc, dataType, symbol, interval)
                            paths_list.append(path)
                    paths[dataType+'.'+symbol+'.'+interval] = paths_list  
                symbolcounter += 1

        return paths


    @staticmethod
    def Unicorn_Market_Channle(dataType, symbol, interval):
        market = str.lower(symbol)
        if dataType == 'klines': 
            channel = 'kline_' + interval
        elif dataType == 'aggTrades':
            channel = 'aggTrade'
        else:
            raise Exception('Unhandled data type: {}'.format(dataType))
        
        return market, channel

    @staticmethod
    def Unicorn_dataType_Symbol_Interval_From_Channel_Market(channel, market):
        symbol = str.upper(market)
        if '_' in channel:
            [dataType, interval] = channel.split('_')
            if dataType == 'kline': dataType = 'klines'
        elif channel == 'aggTrades':
            dataType = 'aggTrade'
            interval = None
        else:
            raise Exception('Unhandled channel: {}'.format(channel))
        
        return dataType, symbol, interval

    @staticmethod
    def Unicorn_dataType_Symbol_Interval(stream_str):
        dataType, symbol, interval = None, None, None

        [market, channel] = stream_str.split('@')
        symbol = str.upper(market)
        if '_' in channel: [dataType, interval] = channel.split('_')
        else: dataType = channel

        if dataType == 'kline': dataType = 'klines'
        
        return dataType, symbol, interval

    def Subscribe_to_Existing_or_New_Active_Stream_v2(self, key):
        # This is a crazy idea. A new channel, so a new thread, for each new key, is allocated.

        [dataType, symbol, interval] = key.split('.')
        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)
        stream_id = Binance.unicorn_sockets_manager.create_stream( channel, market ) # Do not give stream_buffer_name, which defaults to False.
        return stream_id


    def Subscribe_to_Existing_or_New_Active_Stream(self, key):
        #(lock, dynarray, t_filepath, stream_id, tickbuffer, t_state ) = Binance.gstreams[key]
        [dataType, symbol, interval] = key.split('.')
        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)

        stream_shared = False
        # Unicorn, and Binance, creates and allocates one additional thread to each newly created stream.
        # In order to reduce the total number of threads and the overhead of switching between threads, I choose to reuse streams as much as possible.
        # There will be hundreds of markets with the same single channel, for example.

        # with Binance.lock_gstream: # Fast enough, exept the maximum single call to subscrive_to_stream().
        # Did every ugly crooked struggle to avoid "dictionary changed size during iteration" runtime error.
        with Binance.stream_lock:
            stream_ids = None
            try:
                stream_list = copy.deepcopy(Binance.unicorn_sockets_manager.get_stream_list())
                stream_ids = [id for (id, _) in stream_list.items()] # the source of "dictionary size changed while iterating" error.
            except:
                pass    # It works. Even after the "dictionary size changed during iteration", stream_ids are ok.
            # Print("stream_ids: {}".format(stream_ids))

            if stream_ids is not None and isinstance(stream_ids, list) and len(stream_ids) > 0 :
                subscription = None
                # Print("stream_ids : {}".format(stream_ids))
                for stream_id in stream_ids:
                    stream_info = Binance.unicorn_sockets_manager.get_stream_info(stream_id)
                    if stream_info['channels'] is not None and stream_info['markets'] is not None:
                        if len(stream_info['channels']) * len(stream_info['markets']) < Config['max_slots_per_stream']:
                            if len(stream_info['channels']) == 1 and stream_info['channels'][0] == channel :
                                subscription = (stream_id, [], [market]); stream_shared = True
                                # stream_shared = Binance.unicorn_sockets_manager.subscribe_to_stream(stream_id, [], [market])
                            elif len(stream_info['market']) == 1 and stream_info['markets'][0] == market :
                                subscription = (stream_id, [channel], []); stream_shared = True
                                # stream_shared = Binance.unicorn_sockets_manager.subscribe_to_stream(stream_id, [channel], [])

                            if stream_shared:
                                # Print("-----stream shared id={}, channel={}, market={}".format(stream_id, channel, market))
                                break
            
                stream_id = None
                if subscription is not None:
                    Binance.unicorn_sockets_manager.subscribe_to_stream(subscription[0], subscription[1], subscription[2])
                    stream_id = subscription[0]
                else:
                    # if not stream_shared or stream_id is None:
                    stream_id = Binance.unicorn_sockets_manager.create_stream( channel, market ) # Do not give stream_buffer_name, which defaults to False.
                    # Print("---stream created id={}, channel={}, market={}".format(stream_id, channel, market))
            else:
                stream_id = Binance.unicorn_sockets_manager.create_stream( channel, market ) # Do not give stream_buffer_name, which defaults to False.

        return stream_id
    

    def Get_Price_Data_By_Time(self, dataType, symbol, interval, start_utc, end_utc, lookup=False):
        """
        Read price data from start to end. end is EXCLUSIVE.
        Returns: (dynarray, nForged, successful, fetchedFromGstream, reason)
        """
        global file_not_found, no_line_feed
        file_not_found = False
        assert interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']        

        nForged = 0
        successful = True
        reason = ""

        # ========== 1. Convert inputs to UTC and normalize ==========
        if start_utc is None:
            start_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        else:
            start_utc = start_utc.replace(tzinfo=pytz.utc) # from_dt_local_to_dt_utc(start)
        start_utc = min(start_utc, datetime.utcnow().replace(tzinfo=pytz.utc))

        if end_utc is None:
            end_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        else:
            end_utc = end_utc.replace(tzinfo=pytz.utc) # from_dt_local_to_dt_utc(end)
        end_utc = min(end_utc, datetime.utcnow().replace(tzinfo=pytz.utc))

        # Normalize to interval boundaries
        interval_mili = intervalToMilliseconds(interval)
        
        start_timestamp = datetime.timestamp(start_utc)
        end_timestamp = datetime.timestamp(end_utc)
        start_mili = round(start_timestamp * 1000 / interval_mili) * interval_mili
        end_mili = round(end_timestamp * 1000 / interval_mili) * interval_mili
        start_utc = datetime.fromtimestamp(start_mili / 1000, tz=pytz.utc)
        end_utc = datetime.fromtimestamp(end_mili / 1000, tz=pytz.utc)
        
        # Calculate expected rows: from start_mili to end_mili exclusive
        nRowsExpected = int((end_mili - start_mili) / interval_mili)
        if nRowsExpected <= 0: return create_empty_dynarray(), 0, True, False, "No data expected"
        
        start_org = start_utc
        end_org = end_utc
        margin = 2
        start_utc = start_utc - timedelta(days=margin)
        end_utc = end_utc + timedelta(days=margin)
        # end_utc = min(end_utc, datetime.utcnow().replace(tzinfo=pytz.utc))
        
        key = dataType + "." + symbol + "." + interval
        dynarray = create_empty_dynarray()
        slot = get_timeslot(dataType)

        leadMsgShown = False
        leadMsg = "Collecting candles from files ... "

        # ========== 2. Get data from monthly historic files ==========
        end_month_start = get_current_month_start(end_utc)
        month_loops = 0
        focus_month_start = get_current_month_start(start_utc)
        
        success = False
        while focus_month_start <= end_month_start:
            month_loops += 1

            if not leadMsgShown:
                Print(leadMsg, end="")
                leadMsgShown = True
            
            array, fills, success = get_all_from_historic_file(focus_month_start, dataType, symbol, interval, dailyNotMonthly = False)

            if success and array is not None and array.shape[0] > 0:
                data_loaded = True
                nForged += fills
                dynarray = np.append(dynarray, array, axis=0)
            else:
                break

            focus_month_start = get_next_month_start(focus_month_start)
        
        if not success:
            failed_focus_month_start = focus_month_start
            end_day_start = get_current_day_start(end_utc)
            day_loops = 0
            focus_day_start = get_current_day_start(focus_month_start)
            success = False
            while focus_day_start <= end_day_start:
                day_loops += 1
                array, fills, success = get_all_from_historic_file(focus_day_start, dataType, symbol, interval, dailyNotMonthly = True)
                if success and array is not None and array.shape[0] > 0:
                    data_loaded = True
                    nForged += fills
                    dynarray = np.append(dynarray, array, axis=0)
                else:
                    break
                
                focus_day_start = get_next_day_start(focus_day_start)
        
        # ========== 6. Filter to exact requested range ==========
        if data_loaded:
            dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)
            dynarray, nForged_fill, fill_success = fill_rightward(
                self.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval,
                start_mili, end_mili
            )
            nForged += nForged_fill            
            dynarray = select_from_dynarray_by_time(dataType, dynarray, start_org, end_org)
            successful = True
            reason = ""
                    
        else:
            nForged = 0
            dynarray = create_empty_dynarray()
            successful = False
            reason = "No data available from any source"

        print(f"nForged/nTotal/expected {nForged}/{dynarray.shape[0]}/{nRowsExpected}. {reason}")
            
        # Remove timestamp columns (index 0 and 6) for final output
        if dynarray.shape[1] >= 7:
            dynarray = np.delete(dynarray, [0, 6], axis=1)
        dynarray = np.float32(dynarray)
        
        # Do NOT remove this line, as it guide the usage of this module.
        print("Returning 10 columns: O H L C V, Quote V, nTrades, Buy base V, Buy quote V, Mark (0: original not forged)")

        fetchedFromGstream = False
        successful = (dynarray.shape[0] == nRowsExpected)
        if not successful: reason = reason + ". The number of rows mismatches nRowsExpected."

        return dynarray, nForged, successful, fetchedFromGstream, reason

    def GetFeatureList(self):
        return ["Open", "High", "Low", "Close", "Volume", "QuoteVolume", "NumTrades", "BuyBaseVolume", "BuyQuoteVolume", "Mark"]


    def UpdateLookupState(self, firstOpenTimeMile, _latestDataStart, _checkedDataStart, start_lookedup):
        if _latestDataStart is None:
            _latestDataStart = datetime.fromtimestamp(firstOpenTimeMile/1000).replace(tzinfo=pytz.utc)
        if not start_lookedup:
            _checkedDataStart = datetime.fromtimestamp(firstOpenTimeMile/1000).replace(tzinfo=pytz.utc)
            start_lookedup = True
        return _latestDataStart, _checkedDataStart, start_lookedup

    def Place_Order(self):
        return

    def Get_Price_Data_By_Time_Batch(self, dataType, symbols, interval, start, end, lookup=True): # read [start, end]. end INCLUSIVE.
        table = []
        nForged = 0
        successful = True
        nPoints = None
        reports = []
        nSuccess = 0

        Print("Walking through the list...")

        counter = 1
        for symbol in symbols:
            dyarray, _nForged, successful, _, reason = \
            self.Get_Price_Data_By_Time(dataType, symbol, interval, start, end, lookup=lookup)

            report = symbol + ": "
            if successful:
                if nPoints == None:
                    nPoints = dyarray.shape[0]
                elif nPoints != dyarray.shape[0]:
                    successful = False
                    nForged += _nForged

                report += str(dyarray.shape[0]) + ", {} created".format(_nForged)

                if nPoints == dyarray.shape[0]:
                    table.append(dyarray)
                    nForged += _nForged
                    # Print(symbol, end="")
                    report += ". Success. {}-th.".format(counter)
                    nSuccess += 1
                else:
                    report += ". Failure. {}-th. ".format(counter) + reason
                counter += 1

                Print("\n", report)

            reports.append(report)

        if len(table) > 0:
            table = np.stack(table, axis=0)
        else:
            table = np.array([])

        Print("\nSuccessful for {}/{} markets".format(nSuccess, len(symbols)))

        successful = len(symbols) == nSuccess
        return table, nForged, reports, successful


    def Get_Cached_Price(self, dataType, symbol, interval, start):
        array = None
        key = dataType+'.'+symbol+'.'+ interval
        if interval == "1m" and Binance.gstreams.get(key, None) is not None and key not in Binance.badkeys:
            gs = Binance.gstreams[key]
            no_missing = bool(gs.t_state & T_State.no_missing)
            if no_missing:
                from_utc = from_dt_local_to_dt_utc(start)
                array = select_from_dynarray_by_time(dataType, gs.dynarray, from_utc, None)

                # Comment out, as Demissor already did it, and it takes time, and the return data is only for display.
                # if len(array) > 0:
                #     interval_mili = 60000
                #     array = snap_sort_deduplicate_in_opening_time(array, interval_mili)

        return array

    def get_combined_history(self, datatype, interval, start, end, symbol_dict):

        features_to_return, nForged = [], 0

        for symbol in symbol_dict.keys():
            features = symbol_dict[symbol]

            feature_list = self.GetFeatureList()

            candles, nFills, successful, _, reason = self.Get_Price_Data_By_Time(datatype, symbol, interval, start, end)
            print(reason if reason else "No reason provided.")
            if successful:          # price is a numpy array.
                print("Successful !!!!!!!!!! (nCandles, nAttributes) = {}, nForged = {}".format(candles.shape, nFills))
                nForged += nFills

                for feature in features:
                    features_to_return.append(candles[:, feature_list.index(feature)])

        return features_to_return, nForged

    def get_combined_history_2(self, datatype, interval, start, end, symbol_dict, target):

        features_to_return, nForged, target_feature_id = [], 0, None

        target_symbol, target_feature = target

        feature_id = 0
        for symbol in symbol_dict.keys():
            features = symbol_dict[symbol]

            feature_list = self.GetFeatureList()

            candles, nFills, successful, _, reason = self.Get_Price_Data_By_Time(datatype, symbol, interval, start, end)
            print(reason if reason else "No reason provided.")
            if successful:          # price is a numpy array.
                print("Successful !!!!!!!!!! (nCandles, nAttributes) = {}, nForged = {}".format(candles.shape, nFills))
                nForged += nFills

                for feature in features:
                    features_to_return.append(candles[:, feature_list.index(feature)])
                    if symbol == target_symbol and feature == target_feature:
                        target_feature_id = feature_id
                    feature_id += 1

        return features_to_return, nForged, target_feature_id


if __name__ == "__main__":
    
    bn = Binance.instantiate(connect=False)
    
    # Get_Price_Data_By_Time function seamlessly collects/creates price candles from Binance.
    datatype = "klines" # "aggTrades" #, "klines"
    interval = '1m'    # in '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'.  (excludes: '3d', '1w')
    start = datetime(2024, 1, 1)    # Any datetime. Note Binance started operation in July 2017.
    end = datetime(2026, 6, 30) # + timedelta(minutes=100)     # Any datetime

    symbol_dict  = {
        "ETHUSDT": ["Open", "High", "Low", "Close", "Volume"],
        "BTCUSDT": ["Open", "High", "Low", "Close", "Volume"]
    }

    features, nForget = bn.get_combined_history(datatype, interval, start, end, symbol_dict)

    T = features[0].shape[0] if len(features) > 0 else 0
    for feature in features:
        assert feature.shape[0] == T, "All features must have the same number of time points."

    print(f"nFeatures ========== {len(features)}, T ========== {T}, nForged ========== {nForget}")



