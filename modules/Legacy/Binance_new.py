
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
# Add the directory containing this file to Python's path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Utility import * 
from config import *

from ThreadOnTimeMachine import *

from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager as UnicornSockets
import logging

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
    def instantiate(cls, connected=True):
        if cls._singleton is None:

            Binance.Initialize()
            cls.lock_gstream = threading.Lock()
           
            cls._singleton = cls.__new__(cls)
            client = Client(Config['api_key'], Config['api_secret']) #, tld = 'us', testnet = False)
            #client.API_URL = 'https://testnet.binance.vision/api'
            #client = Client()
            cls.client = client #, tld = 'us', testnet = False)

            travel_lag_mili = 150 # Specific to my location. Experimental.
            # This value is chosen so that the following evalustes to be the equal negative value.
            #self.GetExchangeTimeMili() - (datetime.utcnow().replace(tzinfo = pytz.utc)-datetime.utcfromtimestamp(0)).total_seconds() * 1000
            #(datetime.utcnow().replace(tzinfo = pytz.utc)-datetime.utcfromtimestamp(0)).total_seconds() * 1000 - self.GetExchangeTimeMili()

            server_time = client.get_server_time()
            milisec_since_epoch = server_time["serverTime"] + travel_lag_mili
            gmtime=tm.gmtime(milisec_since_epoch/1000.0)
            mili= round(milisec_since_epoch) % 1000
            # win32api.SetSystemTime(gmtime[0], gmtime[1], gmtime[6], gmtime[2], gmtime[3], gmtime[4], gmtime[5], mili) # Needed Administrator Permission

            if connected:
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
            #     gs.dynarray, nCreated, successful = create_missing_prices_inland(self.clientPool, Config['binance_wait_sec'], gs.dynarray, dataType, symbol, interval, start_mili, end_mili + interval_mili)
            #     successful = gs.dynarray.shape[0] == nRowsExpected

            if (round(gs.dynarray[search_end-1, slot]) - round(gs.dynarray[search_start, slot])) == (search_end - 1 - search_start) * interval_mili:
                assert is_one_incremental( gs.dynarray[search_start:search_end, slot].astype(np.int64) / interval_mili ), \
                    "The rows following prepending is not one-incremental."    # time-ordered no-missing.           
            else: # very rare but very expensive
                # Acheives the Goal 2. #######################################
                gs.dynarray, nCreated, no_missing \
                = create_missing_prices_inland(Binance.clientPool, binance_wait_sec, gs.dynarray, dataType, symbol, interval, 0, 0, ignore_last = live and closed)
                if nCreated > 0:
                    monitor += ", F {}".format(nCreated)

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
    
    # def Get_Price_Data_By_Time(self, dataType, symbol, interval, start, end, lookup=False, prePad=False):       # comeback - overriding the below
    #     """
    #     Read price data from start to end. end is INCLUSIVE.
    #     Returns: (dynarray, nCreated, successful, fetchedFromGstream, reason)
    #     """
    #     global file_not_found, no_line_feed
    #     file_not_found = False
    #     assert start is None or start.tzinfo is None
    #     assert end is None or end.tzinfo is None
    #     assert interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        
    #     print("\nGet_Price: {}.{}.{}.{}.{}".format(dataType, symbol, interval, start, end))
    #     start_saved = start

    #     nCreated = 0
    #     successful = True
    #     fetchedFromGstream = False
    #     reason = ""
    #     nRowsExpected = None
    #     start_lookedup = False
    #     data_loaded = False

    #     # ========== 1. Convert inputs to UTC and normalize ==========
    #     if start is None:
    #         start_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    #     else:
    #         start_utc = from_dt_local_to_dt_utc(start)

    #     if end is None:
    #         end_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    #     else:
    #         end_utc = from_dt_local_to_dt_utc(end)
    #     end_utc = min(datetime.utcnow().replace(tzinfo=pytz.utc), end_utc)

    #     # Normalize to interval boundaries
    #     interval_mili = intervalToMilliseconds(interval)
    #     start_utc = datetime.fromtimestamp(
    #         round(datetime.timestamp(start_utc) * 1000 / interval_mili) * interval_mili / 1000
    #     ).replace(tzinfo=pytz.utc)
    #     end_utc = datetime.fromtimestamp(
    #         round(datetime.timestamp(end_utc) * 1000 / interval_mili) * interval_mili / 1000
    #     ).replace(tzinfo=pytz.utc)
        
    #     # If start and end are the same after normalization, we need exactly one candle
    #     # Convert to exclusive end for internal processing, but handle the single-candle case
    #     if start_utc == end_utc:
    #         # Single candle case: end_exclusive should be start + 1 interval
    #         end_utc_exclusive = end_utc + timedelta(milliseconds=interval_mili)
    #     else:
    #         # Normal case: end_exclusive is the next candle after end
    #         end_utc_exclusive = end_utc + timedelta(milliseconds=interval_mili)

    #     key = dataType + "." + symbol + "." + interval
    #     dynarray = create_empty_dynarray()
    #     slot = get_timeslot(dataType)

    #     # ------- work on TokenLookup (commented out in original, kept as placeholder) ------
    #     _checkedFrom = start_utc
    #     _checkedTo = end_utc
    #     _checkedDataStart = None
    #     _latestDataStart = None

    #     if lookup:
    #         # Lookup logic preserved as comments from original
    #         pass

    #     # The start of the day before yesterday
    #     cacheStartTime_utc = get_current_day_start(datetime.utcnow()).replace(tzinfo=pytz.utc) - timedelta(days=1)

    #     leadMsgShown = False
    #     leadMsg = "Collecting candles from files ... : "

    #     # ========== 2. Get data from monthly historic files ==========
    #     monthly_success = False
    #     from_month_start = get_current_month_start(start_utc)
        
    #     while from_month_start < end_utc_exclusive:
    #         from_utc = max(from_month_start, start_utc)
    #         to_utc = min(get_next_month_start(from_month_start) - timedelta(milliseconds=interval_mili), end_utc_exclusive)
            
    #         # Only process if the range is valid and non-empty
    #         if from_utc <= to_utc:
    #             if not leadMsgShown:
    #                 Print(leadMsg, end="")
    #                 leadMsgShown = True
                
    #             array, fills, success = get_from_historic_file(
    #                 self.clientPool, Config['binance_wait_sec'], from_month_start, 
    #                 dataType, symbol, interval, from_utc, to_utc, dailyNotMonthly=False
    #             )
                
    #             if success and array is not None and array.shape[0] > 0:
    #                 monthly_success = True
    #                 data_loaded = True
    #                 nCreated += fills
    #                 dynarray = np.append(dynarray, array, axis=0)
                    
    #         from_month_start = get_next_month_start(from_month_start)

    #     # ========== 3. Get data from daily historic files ==========
    #     daily_success = False
    #     from_day_start = get_current_month_start(cacheStartTime_utc - timedelta(days=2))
    #     from_day_start = max(from_day_start, get_current_day_start(start_utc))
        
    #     while from_day_start < end_utc_exclusive:
    #         from_utc = max(from_day_start, start_utc)
    #         to_utc = min(get_next_day_start(from_day_start) - timedelta(milliseconds=interval_mili), end_utc_exclusive)
            
    #         if from_utc <= to_utc:
    #             if not leadMsgShown:
    #                 Print(leadMsg, end="")
    #                 leadMsgShown = True
                
    #             array, fills, success = get_from_historic_file(
    #                 self.clientPool, Config['binance_wait_sec'], from_day_start,
    #                 dataType, symbol, interval, from_utc, to_utc, dailyNotMonthly=True
    #             )
                
    #             if success and array is not None and array.shape[0] > 0:
    #                 daily_success = True
    #                 data_loaded = True
    #                 nCreated += fills
    #                 dynarray = np.append(dynarray, array, axis=0)
            
    #         from_day_start += timedelta(days=1)

    #     if leadMsgShown:
    #         Print("")  # Newline after progress indicator

    #     # ========== 4. Get data from gstream or online price service ==========
    #     from_utc = max(cacheStartTime_utc, start_utc)
    #     to_utc = end_utc

    #     if from_utc <= end_utc:
    #         from_mili = round(datetime.timestamp(from_utc) * 1000 / interval_mili) * interval_mili
    #         to_mili = round(datetime.timestamp(to_utc) * 1000 / interval_mili) * interval_mili
    #         nRowsExpected = round((to_mili - from_mili) / interval_mili) + 1
            
    #         if nRowsExpected > 0:
    #             # Try gstream first
    #             if interval == "1m" and Binance.gstreams.get(key, None) is not None and key not in Binance.badkeys:
    #                 gs = Binance.gstreams[key]
    #                 no_missing = bool(gs.t_state & T_State.no_missing)
    #                 if no_missing:
    #                     array = select_from_dynarray_by_time(dataType, gs.dynarray, from_utc, end_utc_exclusive)
    #                     if len(array) == nRowsExpected:
    #                         array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
    #                         dynarray = np.append(dynarray, array, axis=0)
    #                         data_loaded = True
    #                         fetchedFromGstream = True
    #                     else:
    #                         self.DemissGStream(key, 0, 0)
                
    #             # If gstream failed or not available, fetch from Binance
    #             if not fetchedFromGstream:
    #                 rows = exact_fetch(
    #                     self.clientPool, Config['binance_wait_sec'], dataType, symbol, interval,
    #                     from_mili, to_mili, nRowsExpected, interval_mili
    #                 )
                    
    #                 if rows is not None:
    #                     if len(rows) == nRowsExpected:
    #                         rows = parse_datalines_to_datalines(dataType, rows)
    #                         array = np.array(rows, dtype=float)
    #                         array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
                            
    #                         if is_one_incremental(array[:, slot].astype(np.int64) / interval_mili):
    #                             dynarray = np.append(dynarray, array, axis=0)
    #                             data_loaded = True
    #                         else:
    #                             Print("??? The rows fetched to fill in is not one-incremental.")
    #                             successful = False
    #                             reason = "Fetched rows are not one-incremental."
    #                     else:
    #                         Print("??? The number of rows fetched to fill in != requested")
    #                         successful = False
    #                         reason = "The number of fetched rows does not meet expectation."

    #     # ========== 5. Deduplicate and sort ==========
    #     if successful and data_loaded and dynarray is not None and dynarray.shape[0] > 0:
    #         # Sort by timestamp and remove duplicates
    #         sort_idx = np.argsort(dynarray[:, slot])
    #         dynarray = dynarray[sort_idx]
            
    #         # Remove duplicate timestamps (keep first occurrence)
    #         _, unique_idx = np.unique(dynarray[:, slot], return_index=True)
    #         if len(unique_idx) < dynarray.shape[0]:
    #             Print(f"????? Removing {dynarray.shape[0] - len(unique_idx)} duplicate rows")
    #             dynarray = dynarray[unique_idx]
            
    #         # Check one-incremental property
    #         if not is_one_incremental(dynarray[:, slot].astype(np.int64) / interval_mili):
    #             Print("????? Final: One_Incremental rule violated. Correcting...")
    #             dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)
        
    #     # ========== 6. Validate and fill missing points ==========
    #     if successful and data_loaded and dynarray is not None and dynarray.shape[0] > 0:
    #         # Calculate expected rows
    #         start_mili = round(datetime.timestamp(start_utc) * 1000 / interval_mili) * interval_mili
    #         end_mili = round(datetime.timestamp(end_utc) * 1000 / interval_mili) * interval_mili
    #         nRowsExpected = round((end_mili - start_mili) / interval_mili) + 1
            
    #         if dynarray.shape[0] < nRowsExpected:
    #             Print("????? Final: Missing points found. Correcting...")
    #             dynarray, nCreated_fill, fill_success = create_missing_prices_inland(
    #                 self.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval,
    #                 start_mili, end_mili + interval_mili
    #             )
    #             nCreated += nCreated_fill
    #             successful = fill_success and (dynarray.shape[0] == nRowsExpected)
                
    #             if not successful:
    #                 reason = "The number of final total rows does not meet expectation."
    #         elif dynarray.shape[0] > nRowsExpected:
    #             # Trim extra rows if we have too many (can happen with boundary overlaps)
    #             Print(f"????? Too many rows: {dynarray.shape[0]} > {nRowsExpected}. Trimming...")
    #             # Keep only rows within the requested range
    #             mask = (dynarray[:, slot] >= start_mili) & (dynarray[:, slot] <= end_mili)
    #             dynarray = dynarray[mask]
    #             if dynarray.shape[0] != nRowsExpected:
    #                 successful = False
    #                 reason = f"After trimming, still have {dynarray.shape[0]} rows, expected {nRowsExpected}"
    #         else:
    #             # Perfect match
    #             pass
    #     elif not data_loaded:
    #         successful = False
    #         if not reason:
    #             reason = "No data available from any source"

    #     # ========== 7. Output results ==========
    #     if successful and dynarray is not None and dynarray.shape[0] > 0:
    #         Print(("\n" if no_line_feed else "") + "nCreated/nTotal/expected {}/{}/{}".format(
    #             nCreated, dynarray.shape[0], nRowsExpected if nRowsExpected else dynarray.shape[0]
    #         ))
    #     else:
    #         Print(("\n" if no_line_feed else "") + "????? FAILURE. {}, nCreated/nTotal/expected {}/{}/{}".format(
    #             symbol, nCreated, dynarray.shape[0] if dynarray is not None else 0, 
    #             nRowsExpected if nRowsExpected else "N/A"
    #         ))

    #     # ========== 8. Pre-pad if requested ==========
    #     if successful and dynarray is not None and dynarray.shape[0] > 0 and prePad and start_saved is not None:
    #         Print("Extrapolated candles will be created if initial candles are not available...", end="")
    #         dynarray, _nCreated, prepad_success = PrePad_with_Zero(
    #             Binance.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval,
    #             interval_mili, start_saved.replace(tzinfo=pytz.utc)
    #         )
    #         nCreated += _nCreated
    #         successful = prepad_success

    #     # ========== 9. Clean up and convert to float32 ==========
    #     if dynarray is not None and dynarray.shape[0] > 0:
    #         # Clean up any placeholder values
    #         if dynarray.dtype == np.dtype('O'):  # Object type
    #             dynarray[dynarray == '12.12.03000000'] = 0.0
            
    #         # Remove timestamp columns (index 0 and 6) for final output
    #         # Only if we have at least 7 columns (indices 0 and 6 exist)
    #         if dynarray.shape[1] >= 7:
    #             dynarray = np.delete(dynarray, [0, 6], axis=1)
    #         dynarray = np.float32(dynarray)

    #     return dynarray, nCreated, successful, fetchedFromGstream, reason


    # def Get_Price_Data_By_Time(self, dataType, symbol, interval, start, end, lookup = False, prePad = False ): # read [start, end]. end INCLUSIVE.
    #     global file_not_found
    #     file_not_found = False
    #     assert start is None or start.tzinfo == None
    #     assert end is None or end.tzinfo == None
    #     assert interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'] #, '3d', '1w']
    #     print("\nGet_Price: {}.{}.{}.{}.{}".format(dataType, symbol, interval, start, end))
    #     start_saved = start

    #     nCreated = 0
    #     successful = True
    #     fetchedFromGstream = False
    #     reason = ""
    #     nRowsExpected = None
    #     start_lookedup = False

    #     #========== 1. Get data from historic files 
    #     # -----  Read (-infinity, today_start_utc) intersection [start_utc, end_utc), if not empty.

    #     if start is None:
    #         start_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    #     else:
    #         start_utc = from_dt_local_to_dt_utc( start )

    #     if end is None: 
    #         end_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    #     else:
    #         end_utc = from_dt_local_to_dt_utc( end )
    #     end_utc = min(datetime.utcnow().replace(tzinfo=pytz.utc), end_utc )

    #     # This may cause the two to get closer, or even to meet.
    #     # if Binance.gstream_datetime_utc is not None:
    #     #     start_utc = min(start_utc, Binance.gstream_datetime_utc)
    #     #     end_utc = min(end_utc, Binance.gstream_datetime_utc)

    #     # Normalize. round is safe?
    #     interval_mili = intervalToMilliseconds(interval)
    #     start_utc = datetime.fromtimestamp( round(datetime.timestamp(start_utc)*1000/interval_mili) * round(interval_mili/1000)).replace(tzinfo=pytz.utc)
    #     end_utc = datetime.fromtimestamp(round(datetime.timestamp(end_utc)*1000/interval_mili) * round(interval_mili/1000)).replace(tzinfo=pytz.utc)

    #     key = dataType + "." + symbol + "." + interval
    #     dynarray = create_empty_dynarray()
    #     slot = get_timeslot(dataType)

    #     # if Binance.gstreams.get(key, None) is None:
    #     #     Print("{} is not available from Binance.".format(key))
    #     #     successful = False
    #     #     fetchedFromGstream = False
    #     #     reason = "Not available from Binance"
    #     #     return dynarray, nCreated, successful, fetchedFromGstream, reason

    #     # ------- work on TokenLookup ----------------------
    #     _checkedFrom = start_utc; _checkedTo = end_utc
    #     _checkedDataStart = None
    #     _latestDataStart = None

    #     if lookup:
    #         pass
    #         # if Binance.tokenLookup.get(key, None) is not None:
    #         #     value = Binance.tokenLookup.get(key)
    #         #     checkedFrom = value["checkedFrom"]
    #         #     checkedTo = value["checkedTo"]
    #         #     checkedDataStart = value["checkedDataStart"]
    #         #     _checkedFrom = min(checkedFrom, start_utc)  # widen the info
    #         #     _checkedTo = max(checkedTo, end_utc)        # widen the info
                
    #         #     if checkedDataStart != None and checkedDataStart <= start_utc:   # we will have no chance to update checkedDataStart.
    #         #         _checkedDataStart = checkedDataStart
    #         #         start_lookedup = True   # to prevent _checkedDataStart from changing.

    #         #     if checkedFrom != None and checkedDataStart != None and checkedFrom < checkedDataStart:  # checkedDataStart is final and we should not change it
    #         #         _checkedDataStart = checkedDataStart
    #         #         start_lookedup = True   # to prevent _checkedDataStart from changing.

    #         #         if start_utc <= checkedDataStart:
    #         #             start_utc = checkedDataStart
    #     start_lookedup = False

    #     # The start of the day before yesterday.
    #     cacheStartTime_utc =  get_current_day_start(datetime.utcnow()).replace(tzinfo=pytz.utc) - timedelta(days=1)  

    #     from_utc = None; to_utc = None

    #     leadMsgShown = False
    #     leadMsg = "Collecting candles from files ... : "
    #     success0 = False; success1 = False
    #     from_month_start = get_current_month_start(start_utc)
    #     while not (success0 and not success1) and from_month_start < end_utc: # get_current_month_start(cacheStartTime_utc - timedelta(days=2)):
    #         success0 = success1
    #         # Ok, it's not the same month as cacheStartTime_utc.
    #         from_utc = max( from_month_start, start_utc )
    #         to_utc = min( get_next_month_start(from_month_start) - timedelta(milliseconds=interval_mili), end_utc)
    #         # Thus, [from_utc, to_utc) = [sample_utc's month]  intersection [start_utc, end_utc]
    #         # Price file for this month should exist unless there were no transactions for the month.
    #         # Now, should we really download the monthly data, which is 30 times bigger than daily one?

    #         if from_utc <= to_utc:
    #             if not leadMsgShown:
    #                 Print(leadMsg, end="")
    #                 leadMsgShown = True
    #             array, fills, success1 = get_from_historic_file(self.clientPool, Config['binance_wait_sec'], from_month_start, dataType, symbol, interval, from_utc, to_utc, dailyNotMonthly = False) # read [from_utc, to_utc)
    #             if success1: # and lookup:
    #                 nCreated += fills
    #                 dynarray = np.append(dynarray, array, axis=0)
    #                 # _latestDataStart, _checkedDataStart, start_lookedup \
    #                 #     = self.UpdateLookupState(dynarray[0, 0], _latestDataStart, _checkedDataStart, start_lookedup)
                        
    #             if not success1:
    #                 Print(f"get_from_historic_file failed at {from_month_start}, {from_utc}, {end_utc}")
                    
    #         from_month_start = get_next_month_start(from_month_start)

    #     from_day_start = get_current_month_start(cacheStartTime_utc - timedelta(days=2))
    #     from_day_start = max(from_day_start, get_current_day_start(start_utc))
    #     while not (success0 and not success1) and from_day_start < end_utc: # cacheStartTime_utc:  # ASSUMPTION: cacheStartTime_utc is the start of a day.
    #         success0 = success1
    #         from_utc = max( from_day_start, start_utc )
    #         to_utc = min( get_next_day_start(from_day_start) - timedelta(milliseconds=interval_mili), end_utc)
    #         # Thus, [from_utc, to_utc) = [sample_utc's day]  intersection [start_utc, end_utc]
    #         if from_utc <= to_utc:
    #             if not leadMsgShown:
    #                 Print(leadMsg, end="")
    #                 leadMsgShown = True

    #             array, fills, success1 = get_from_historic_file(self.clientPool, Config['binance_wait_sec'], from_day_start, dataType, symbol, interval, from_utc, to_utc) # read [from_utc, to_utc)
    #             if success1:
    #                 nCreated += fills
    #                 dynarray = np.append(dynarray, array, axis=0)
    #                 # _latestDataStart, _checkedDataStart, start_lookedup \
    #                 #     = self.UpdateLookupState(dynarray[0, 0], _latestDataStart, _checkedDataStart, start_lookedup)

    #         from_day_start += timedelta(days=1)

    #     # if leadMsgShown:
    #     #     Print("\n... done with collecting.")

    #     #========== 2. Get data from the gstream or online price service
    #     # ----- Read [cacheStartTime_utc, the last closing candle] intersection [start_utx, end_utc)
    #     fetchedFromGstream = False
    #     if not (success0 and not success1):
    #         from_utc = max(cacheStartTime_utc, start_utc)
    #         to_utc = end_utc

    #         if from_utc <= end_utc:
    #             from_mili= round(datetime.timestamp(from_utc)*1000/interval_mili) * interval_mili
    #             to_mili= round(datetime.timestamp(to_utc)*1000/interval_mili) * interval_mili # inclusive
    #             nRowsExpected = round((to_mili - from_mili) / interval_mili) + 1
                
    #             if nRowsExpected > 0:
    #                 key = dataType+'.'+symbol+'.'+ interval
    #                 if interval == "1m" and Binance.gstreams.get(key, None) is not None and key not in Binance.badkeys:
    #                     gs = Binance.gstreams[key]
    #                     no_missing = bool(gs.t_state & T_State.no_missing)
    #                     if no_missing:
    #                         array = select_from_dynarray_by_time(dataType, gs.dynarray, from_utc, to_utc) # read [from_utc, to_utc]
    #                         if len(array) == nRowsExpected:
    #                             array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
    #                             dynarray = np.append(dynarray, array, axis=0)
    #                             # _latestDataStart, _checkedDataStart, start_lookedup \
    #                             #     = self.UpdateLookupState(dynarray[0, 0], _latestDataStart, _checkedDataStart, start_lookedup)
    #                             fetchedFromGstream = True
    #                         else:
    #                             self.DemissGStream(key, 0, 0)
    #                             pass # not  successful = False


    #                 if not fetchedFromGstream:
    #                     rows = exact_fetch(self.clientPool, Config['binance_wait_sec'], dataType, symbol, interval, from_mili, to_mili, nRowsExpected, interval_mili)
    #                     if rows is not None:
    #                         # Acheives the Goal 2. #######################################
    #                         if len(rows) == nRowsExpected:
    #                             rows = parse_datalines_to_datalines(dataType, rows) # Convert string to float.
    #                             array = np.array(rows) # dtype should be float64.
    #                             array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
    #                             if is_one_incremental( array[:, slot].astype(np.int64) / interval_mili ):
    #                                 dynarray = np.append(dynarray, array, axis=0)
    #                                 # _latestDataStart, _checkedDataStart, start_lookedup \
    #                                 #     = self.UpdateLookupState(dynarray[0, 0], _latestDataStart, _checkedDataStart, start_lookedup)
    #                             else:
    #                                 Print("??? The rows fetched to fill in is not one-incremental.")    # time-ordered no-missing.
    #                                 successful = False
    #                                 reason = "Fetched rows are not one-incremental."
    #                         else:
    #                             Print("??? The number of rows fetched to fill in != requested")
    #                             successful = False
    #                             reason = "The number of fetched rows does not meet expectation."
    #     else:
    #         successful = False

    #     if successful: # and (_latestDataStart is not None):
    #         if not is_one_incremental( dynarray[:, slot].astype(np.int64) / interval_mili ):
    #             Print("????? Final: One_Incremental rule violated. Correcting...")
    #             dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)   # This is a cross-file check.

    #         # start_mili= round(datetime.timestamp(_latestDataStart)*1000/interval_mili) * interval_mili
    #         start_mili= round(datetime.timestamp(start_utc)*1000/interval_mili) * interval_mili
    #         end_mili= round(datetime.timestamp(end_utc)*1000/interval_mili) * interval_mili
    #         nRowsExpected= round((end_mili - start_mili) / interval_mili) + 1

    #         if dynarray.shape[0] < nRowsExpected:
    #             # Print("??? Missing data points found in the file: {}/{}. Filling in ... ".format(dynarray.shape[0], nRowsExpected))
    #             Print("????? Final: Missing points found. Correcting...")
    #             dynarray, nCreated, successful = create_missing_prices_inland(self.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval, start_mili, end_mili + interval_mili)
    #             # Print("Done." if successful else "Failed.")
    #             successful = dynarray.shape[0] == nRowsExpected
    #         elif dynarray.shape[0] > nRowsExpected:
    #             assert False
    #         else:
    #             pass
            
    #         if not successful:
    #             reason = "The number of final total rows does not meet expectation."

    #     global no_line_feed
    #     if successful and (dynarray is not None):
    #         Print(("\n" if no_line_feed else "") + "nCreated/nTotal/expected {}/{}/{}".format(nCreated, dynarray.shape[0], nRowsExpected))
    #         # if lookup:
    #         #     value = Binance.tokenLookup.get(key, {})
    #         #     value.update( {
    #         #         "checkedFrom": _checkedFrom, 
    #         #         "checkedTo": _checkedTo, 
    #         #         "checkedDataStart": _checkedDataStart, 
    #         #         "integral": "Yes",
    #         #         "latestDataStart": _latestDataStart,
    #         #         "latestExpected": dynarray.shape[0], 
    #         #         "latestMissing": nCreated, 
    #         #         } )
    #         #     Binance.tokenLookup[key] = value
    #         #     updateTokenLookup(Binance.tokenLookup)

    #     else:
    #         Print(("\n" if no_line_feed else "") + "????? FAILURE. {}, nCreated/nTotal/expected {}/{}/{}".format(symbol, nCreated, dynarray.shape[0], nRowsExpected))
    #         # if lookup:
    #         #     value = Binance.tokenLookup.get(key, {})
    #         #     value.update( {
    #         #         "checkedFrom": _checkedFrom, 
    #         #         "checkedTo": _checkedTo, 
    #         #         "checkedDataStart": _checkedDataStart, 
    #         #         "integral": "No"
    #         #         })
    #         #     Binance.tokenLookup[key] = value
    #         #     updateTokenLookup(Binance.tokenLookup)

    #     if successful and dynarray.shape[0] > 0 and prePad and start_saved != None:
    #         Print("Extrapolated candles will be created if initial candles are not available...", end="")
    #         dynarray, _nCreated, successful = \
    #             PrePad_with_Zero(Binance.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval, interval_mili, start_saved.replace(tzinfo=pytz.utc))
    #         nCreated += _nCreated

    #     dynarray[dynarray == '12.12.03000000'] = 0.0    # AliceUSDT.3m

    #     if dynarray is not None: 
    #         dynarray = np.delete(dynarray, [0, 6], axis=1) # timestamp (in second or milisecond) prohibits downsizing to float32.
    #         dynarray = np.float32(dynarray)

    #     return dynarray, nCreated, successful, fetchedFromGstream, reason


    # def Get_Price_Data_By_Time(self, dataType, symbol, interval, start, end, lookup=False, prePad=False):   # comeback - overridingg the above
    #     """
    #     Read price data from start to end. end is INCLUSIVE.
    #     Returns: (dynarray, nCreated, successful, fetchedFromGstream, reason)
    #     """
    #     global file_not_found, no_line_feed
    #     file_not_found = False
    #     assert start is None or start.tzinfo is None
    #     assert end is None or end.tzinfo is None
    #     assert interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        
    #     print("\nGet_Price: {}.{}.{}.{}.{}".format(dataType, symbol, interval, start, end))
    #     start_saved = start

    #     nCreated = 0
    #     successful = True
    #     fetchedFromGstream = False
    #     reason = ""
    #     nRowsExpected = None
    #     start_lookedup = False
    #     data_loaded = False

    #     # ========== 1. Convert inputs to UTC and normalize ==========
    #     if start is None:
    #         start_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    #     else:
    #         start_utc = from_dt_local_to_dt_utc(start)

    #     if end is None:
    #         end_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    #     else:
    #         end_utc = from_dt_local_to_dt_utc(end)
    #     end_utc = min(datetime.utcnow().replace(tzinfo=pytz.utc), end_utc)

    #     # Normalize to interval boundaries
    #     interval_mili = intervalToMilliseconds(interval)
    #     interval_sec = interval_mili / 1000
        
    #     start_timestamp = datetime.timestamp(start_utc)
    #     end_timestamp = datetime.timestamp(end_utc)
        
    #     start_mili_normalized = round(start_timestamp * 1000 / interval_mili) * interval_mili
    #     end_mili_normalized = round(end_timestamp * 1000 / interval_mili) * interval_mili
        
    #     start_utc = datetime.fromtimestamp(start_mili_normalized / 1000).replace(tzinfo=pytz.utc)
    #     end_utc = datetime.fromtimestamp(end_mili_normalized / 1000).replace(tzinfo=pytz.utc)
        
    #     # For internal processing, we use exclusive end (one interval after the last inclusive candle)
    #     # But careful: if end_utc is exactly at a candle boundary, the last candle is at end_utc
    #     end_utc_exclusive = end_utc + timedelta(milliseconds=interval_mili)

    #     key = dataType + "." + symbol + "." + interval
    #     dynarray = create_empty_dynarray()
    #     slot = get_timeslot(dataType)

    #     # ------- work on TokenLookup (commented out in original, kept as placeholder) ------
    #     _checkedFrom = start_utc
    #     _checkedTo = end_utc
    #     _checkedDataStart = None
    #     _latestDataStart = None

    #     if lookup:
    #         # Lookup logic preserved as comments from original
    #         pass

    #     # The start of the day before yesterday
    #     cacheStartTime_utc = get_current_day_start(datetime.utcnow()).replace(tzinfo=pytz.utc) - timedelta(days=1)

    #     leadMsgShown = False
    #     leadMsg = "Collecting candles from files ... : "

    #     # ========== 2. Get data from monthly historic files ==========
    #     from_month_start = get_current_month_start(start_utc)
        
    #     while from_month_start < end_utc_exclusive:
    #         # Calculate the month's end (exclusive)
    #         month_end_exclusive = get_next_month_start(from_month_start)
            
    #         # The range to read from this month file
    #         from_utc = max(from_month_start, start_utc)
    #         to_utc = min(month_end_exclusive - timedelta(milliseconds=interval_mili), end_utc_exclusive)
            
    #         # Only process if the range is valid and non-empty
    #         if from_utc <= to_utc:
    #             if not leadMsgShown:
    #                 Print(leadMsg, end="")
    #                 leadMsgShown = True
                
    #             array, fills, success = get_from_historic_file(
    #                 self.clientPool, Config['binance_wait_sec'], from_month_start, 
    #                 dataType, symbol, interval, from_utc, to_utc, dailyNotMonthly=False
    #             )
                
    #             if success and array is not None and array.shape[0] > 0:
    #                 data_loaded = True
    #                 nCreated += fills
    #                 dynarray = np.append(dynarray, array, axis=0)
                    
    #         from_month_start = get_next_month_start(from_month_start)

    #     # ========== 3. Get data from daily historic files ==========
    #     from_day_start = get_current_month_start(cacheStartTime_utc - timedelta(days=2))
    #     from_day_start = max(from_day_start, get_current_day_start(start_utc))
        
    #     while from_day_start < end_utc_exclusive:
    #         # Calculate the day's end (exclusive)
    #         day_end_exclusive = get_next_day_start(from_day_start)
            
    #         # The range to read from this day file
    #         from_utc = max(from_day_start, start_utc)
    #         to_utc = min(day_end_exclusive - timedelta(milliseconds=interval_mili), end_utc_exclusive)
            
    #         if from_utc <= to_utc:
    #             if not leadMsgShown:
    #                 Print(leadMsg, end="")
    #                 leadMsgShown = True
                
    #             array, fills, success = get_from_historic_file(
    #                 self.clientPool, Config['binance_wait_sec'], from_day_start,
    #                 dataType, symbol, interval, from_utc, to_utc, dailyNotMonthly=True
    #             )
                
    #             if success and array is not None and array.shape[0] > 0:
    #                 data_loaded = True
    #                 nCreated += fills
    #                 dynarray = np.append(dynarray, array, axis=0)
            
    #         from_day_start += timedelta(days=1)

    #     if leadMsgShown:
    #         Print("")  # Newline after progress indicator

    #     # ========== 4. Get data from gstream or online price service ==========
    #     from_utc = max(cacheStartTime_utc, start_utc)
    #     to_utc = end_utc  # Inclusive end

    #     if from_utc <= end_utc:
    #         from_mili = round(datetime.timestamp(from_utc) * 1000 / interval_mili) * interval_mili
    #         to_mili = round(datetime.timestamp(to_utc) * 1000 / interval_mili) * interval_mili
    #         # Expected rows: from start_mili to end_mili inclusive
    #         nRowsExpected = int((to_mili - from_mili) / interval_mili) + 1
            
    #         if nRowsExpected > 0:
    #             # Try gstream first
    #             if interval == "1m" and Binance.gstreams.get(key, None) is not None and key not in Binance.badkeys:
    #                 gs = Binance.gstreams[key]
    #                 no_missing = bool(gs.t_state & T_State.no_missing)
    #                 if no_missing:
    #                     # Use exclusive end for selection
    #                     array = select_from_dynarray_by_time(dataType, gs.dynarray, from_utc, end_utc_exclusive)
    #                     if len(array) == nRowsExpected:
    #                         array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
    #                         dynarray = np.append(dynarray, array, axis=0)
    #                         data_loaded = True
    #                         fetchedFromGstream = True
    #                     else:
    #                         self.DemissGStream(key, 0, 0)
                
    #             # If gstream failed or not available, fetch from Binance
    #             if not fetchedFromGstream:
    #                 rows = exact_fetch(
    #                     self.clientPool, Config['binance_wait_sec'], dataType, symbol, interval,
    #                     from_mili, to_mili, nRowsExpected, interval_mili
    #                 )
                    
    #                 if rows is not None:
    #                     if len(rows) == nRowsExpected:
    #                         rows = parse_datalines_to_datalines(dataType, rows)
    #                         array = np.array(rows, dtype=float)
    #                         array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
                            
    #                         if is_one_incremental(array[:, slot].astype(np.int64) / interval_mili):
    #                             dynarray = np.append(dynarray, array, axis=0)
    #                             data_loaded = True
    #                         else:
    #                             Print("??? The rows fetched to fill in is not one-incremental.")
    #                             successful = False
    #                             reason = "Fetched rows are not one-incremental."
    #                     else:
    #                         Print(f"??? The number of rows fetched ({len(rows)}) != requested ({nRowsExpected})")
    #                         successful = False
    #                         reason = "The number of fetched rows does not meet expectation."

    #     # ========== 5. Deduplicate and sort ==========
    #     if successful and data_loaded and dynarray is not None and dynarray.shape[0] > 0:
    #         # Sort by timestamp
    #         sort_idx = np.argsort(dynarray[:, slot])
    #         dynarray = dynarray[sort_idx]
            
    #         # Remove duplicate timestamps (keep first occurrence)
    #         _, unique_idx = np.unique(dynarray[:, slot], return_index=True)
    #         if len(unique_idx) < dynarray.shape[0]:
    #             Print(f"????? Removing {dynarray.shape[0] - len(unique_idx)} duplicate rows")
    #             dynarray = dynarray[unique_idx]
            
    #         # Check one-incremental property
    #         time_ratios = dynarray[:, slot].astype(np.int64) / interval_mili
    #         if not is_one_incremental(time_ratios):
    #             Print("????? Final: One_Incremental rule violated. Correcting...")
    #             dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)
        
    #     # ========== 6. Validate and fill missing points ==========
    #     if successful and data_loaded and dynarray is not None and dynarray.shape[0] > 0:
    #         # Calculate expected rows for the requested range
    #         start_mili = int(round(datetime.timestamp(start_utc) * 1000 / interval_mili) * interval_mili)
    #         end_mili = int(round(datetime.timestamp(end_utc) * 1000 / interval_mili) * interval_mili)
    #         nRowsExpected = int((end_mili - start_mili) / interval_mili) + 1
            
    #         # Filter to only the requested time range
    #         mask = (dynarray[:, slot] >= start_mili) & (dynarray[:, slot] <= end_mili)
    #         dynarray = dynarray[mask]
            
    #         if dynarray.shape[0] < nRowsExpected:
    #             Print(f"????? Final: Missing points found. Have {dynarray.shape[0]}, expected {nRowsExpected}. Filling...")
    #             dynarray, nCreated_fill, fill_success = fill_inward(
    #                 self.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval,
    #                 start_mili, end_mili + interval_mili
    #             )
    #             nCreated += nCreated_fill
    #             successful = fill_success and (dynarray.shape[0] == nRowsExpected)
                
    #             if not successful:
    #                 reason = f"After filling: have {dynarray.shape[0]}, expected {nRowsExpected}"
    #         elif dynarray.shape[0] > nRowsExpected:
    #             # This shouldn't happen after dedup and filtering, but handle gracefully
    #             Print(f"????? Too many rows: {dynarray.shape[0]} > {nRowsExpected}")
    #             # Keep only the first nRowsExpected rows (assuming they're the correct ones)
    #             if dynarray.shape[0] > nRowsExpected:
    #                 dynarray = dynarray[:nRowsExpected]
    #                 successful = (dynarray.shape[0] == nRowsExpected)
    #                 if not successful:
    #                     reason = f"After trimming: have {dynarray.shape[0]}, expected {nRowsExpected}"
    #         else:
    #             # Perfect match
    #             pass
    #     elif not data_loaded:
    #         successful = False
    #         if not reason:
    #             reason = "No data available from any source"

    #     # ========== 7. Output results ==========
    #     if successful and dynarray is not None and dynarray.shape[0] > 0:
    #         Print(("\n" if no_line_feed else "") + "nCreated/nTotal/expected {}/{}/{}".format(
    #             nCreated, dynarray.shape[0], nRowsExpected if nRowsExpected else dynarray.shape[0]
    #         ))
    #     else:
    #         Print(("\n" if no_line_feed else "") + "????? FAILURE. {}, nCreated/nTotal/expected {}/{}/{}".format(
    #             symbol, nCreated, dynarray.shape[0] if dynarray is not None else 0, 
    #             nRowsExpected if nRowsExpected else "N/A"
    #         ))

    #     # ========== 8. Pre-pad if requested ==========
    #     if successful and dynarray is not None and dynarray.shape[0] > 0 and prePad and start_saved is not None:
    #         Print("Extrapolated candles will be created if initial candles are not available...", end="")
    #         dynarray, _nCreated, prepad_success = PrePad_with_Zero(
    #             Binance.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval,
    #             interval_mili, start_saved.replace(tzinfo=pytz.utc)
    #         )
    #         nCreated += _nCreated
    #         successful = prepad_success

    #     # ========== 9. Clean up and convert to float32 ==========
    #     if dynarray is not None and dynarray.shape[0] > 0:
    #         # Clean up any placeholder values
    #         if dynarray.dtype == np.dtype('O'):  # Object type
    #             dynarray[dynarray == '12.12.03000000'] = 0.0
            
    #         # Remove timestamp columns (index 0 and 6) for final output
    #         if dynarray.shape[1] >= 7:
    #             dynarray = np.delete(dynarray, [0, 6], axis=1)
    #         dynarray = np.float32(dynarray)

    #     return dynarray, nCreated, successful, fetchedFromGstream, reason

    def Get_Price_Data_By_Time(self, dataType, symbol, interval, start_utc, end_utc, lookup=False, prePad=False):
        """
        Read price data from start to end. end is EXCLUSIVE.
        Returns: (dynarray, nCreated, successful, fetchedFromGstream, reason)
        """
        global file_not_found, no_line_feed
        file_not_found = False
        assert interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']        

        nCreated = 0
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
                nCreated += fills
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
                    nCreated += fills
                    dynarray = np.append(dynarray, array, axis=0)
                else:
                    break
                
                focus_day_start = get_next_day_start(focus_day_start)
        
        # ========== 6. Filter to exact requested range ==========
        if data_loaded:
            dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)
            dynarray, nCreated_fill, fill_success = fill_rightward(
                self.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval,
                start_mili, end_mili
            )
            nCreated += nCreated_fill            
            dynarray = select_from_dynarray_by_time(dataType, dynarray, start_org, end_org)
            successful = True
            reason = ""
                    
        else:
            nCreated = 0
            dynarray = create_empty_dynarray()
            successful = False
            reason = "No data available from any source"

        print(f"nCreated/nTotal/expected {nCreated}/{dynarray.shape[0]}/{nRowsExpected}. {reason}")
            
        # Remove timestamp columns (index 0 and 6) for final output
        if dynarray.shape[1] >= 7:
            dynarray = np.delete(dynarray, [0, 6], axis=1)
        dynarray = np.float32(dynarray)
        
        # Do NOT remove this line, as it guide the usage of this module.
        print("Returning 10 columns: O H L C V, Quote V, nTrades, Buy base V, Buy quote V, Mark (0: original not forged)")

        fetchedFromGstream = False
        successful = (dynarray.shape[0] == nRowsExpected)
        if not successful: reason = reason + ". The number of rows mismatches nRowsExpected."

        return dynarray, nCreated, successful, fetchedFromGstream, reason

    def UpdateLookupState(self, firstOpenTimeMile, _latestDataStart, _checkedDataStart, start_lookedup):
        if _latestDataStart is None:
            _latestDataStart = datetime.fromtimestamp(firstOpenTimeMile/1000).replace(tzinfo=pytz.utc)
        if not start_lookedup:
            _checkedDataStart = datetime.fromtimestamp(firstOpenTimeMile/1000).replace(tzinfo=pytz.utc)
            start_lookedup = True
        return _latestDataStart, _checkedDataStart, start_lookedup

    def Place_Order(self):
        return

    def Get_Price_Data_By_Time_Batch(self, dataType, symbols, interval, start, end, lookup=True, prePad=True): # read [start, end]. end INCLUSIVE.
        table = []
        nCreated = 0
        successful = True
        nPoints = None
        reports = []
        nSuccess = 0

        Print("Walking through the list...")

        counter = 1
        for symbol in symbols:
            dyarray, _nCreated, successful, _, reason = \
            self.Get_Price_Data_By_Time(dataType, symbol, interval, start, end, lookup=lookup, prePad=prePad)

            report = symbol + ": "
            if successful:
                if nPoints == None:
                    nPoints = dyarray.shape[0]
                elif nPoints != dyarray.shape[0]:
                    successful = False
                    nCreated += _nCreated

                report += str(dyarray.shape[0]) + ", {} created".format(_nCreated)

                if nPoints == dyarray.shape[0]:
                    table.append(dyarray)
                    nCreated += _nCreated
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
        return table, nCreated, reports, successful


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


if __name__ == "__main__":
    
    bn = Binance.instantiate()
    
    # Get_Price_Data_By_Time function seamlessly collects/creates price candles from Binance.
    datatype = "klines" # "aggTrades" #, "klines"
    symbol = "BTCUSDT"      # Any pair listed on Binance.

    # dates that worked
    # start = datetime(2018, 1, 1)    # Any datetime. Note Binance started operation in July 2017.
    # end = datetime(2026, 3, 26) # + timedelta(minutes=100)     # Any datetime

    start = datetime(2020, 1, 1)    # Any datetime. Note Binance started operation in July 2017.
    end = datetime(2025, 3, 26) # + timedelta(minutes=100)     # Any datetime

    interval = '1m'    # in '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'.  (excludes: '3d', '1w')
    print(f"interval: {interval}")
    prePad = True           # prepend with null candles.
    #------------------------------------------------------------------------------------------------------------------
    candles, nFills, successful, _, reason = bn.Get_Price_Data_By_Time(datatype, symbol, interval, start, end, prePad=prePad)
    print(reason)
    if successful:          # price is a numpy array.
        print("\n Successful !!!!!!!!!! (nCandles, nAttributes) = {}, nCreated = {}".format(candles.shape, nFills))