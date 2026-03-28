
import os, sys
from datetime import datetime, timedelta
import pytz
from enum import Enum
import time as tm

Config = {}

# intervals: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'.

Config['ginterval'] = '12h'

def getIntervalMin(interval):
    ret = None
    num = int(interval[:-1])
    unit = interval[-1]
    if unit == 'm': ret = num
    elif unit == 'h': ret = num * 60
    elif unit == 'd': ret = num * 60 * 24
    else: raise Exception('Unknown unit')
    return ret

BasicIntervalMin = getIntervalMin(Config['ginterval'])

Config['start_time_utc'] = datetime.utcnow().replace(tzinfo = pytz.utc)
 
Root = "D:/lagacy_trade_data"
Config['Root'] = Root

if sys.platform.startswith('linux'):
    Config['Data'] = os.path.join('/mnt/data', 'Trading')
elif  sys.platform =='win32':
    Config['Data'] = "D://trade_data/binance_data"
 
Config['History'] = os.path.join( Root, 'history' )
Config['pathMarketLookup'] = os.path.join(Config['Data'], "lookup", "marketLookup_{}.json".format(Config['ginterval']))
Config['BinanceCandlesNumpy'] = os.path.join(Config['Data'], 'candles')

os.environ["STORE_DIRECTORY"] = Config['Data']

Config['.NCheck'] = 30

Config['api_key'] = os.getenv('Binance_API_Key')
Config['api_secret'] = os.getenv('Binance_API_Secret')

Config['dateformat'] = "%b %d, %Y"
Config['datefimeformat'] = "%b %d, %Y %H:%M:%S"

Config['binance_wait_sec'] = 3.0
Config['max_slots_per_stream'] = 50 # The larger it is, the more requests per stream. The smaller it is, the mare stream threads we have.
Config['exchange'] = "binance.com" # https://pypi.org/project/unicorn-binance-websocket-api/
Config['max_subscriptions_per_stream'] = 200    # keep it large to reduce the number of threads.
        # code in unicorn manager:
        # if self.exchange == "binance.com":
        #       self.websocket_base_uri = "wss://stream.binance.com:9443/"
        #       self.max_subscriptions_per_stream = 1024
Config['timeSnapSlippageSec'] = 10

# tm.sleep(
#     Config['Bias_Sec_per_Demiss'] / nGStreams
#     + Config['Linear_Sec_per_Lag_X_#gstream'] * Binance.gstream_lag_sec * nGStreams
# )
Config['Bias_Sec_per_Demiss'] = 60  # so, if there is no lagging, this is the cycle of demissing.
Config['Linear_Sec_per_Lag_X_#gstream'] = 0.01  # if 100 gstreams are 1 second lagging, then 1 second more sleeping.

import threading
lockPrint = threading.Lock()


import logging
logging.basicConfig(
    filename="output_mine.log",
    filemode="w",
    level=logging.CRITICAL,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %I:%M:%S%p",
)

def Print(*argv, end="\n"):
    str = ''
    for arg in argv:
        str += arg
    lockPrint.acquire()
    print(str, end=end)
    lockPrint.release()

class Tense(Enum):
    Past = 1
    Present = 2
    Future = 3
#-------------------------- Controls
Config['Stream_Price'] = True   # Tense.Past will always not stream.
Config['Show_Transceiver'] = False
Config['Run_Demisser'] = True
Config['Show_Demisser'] = True
Config['Run_Traders'] = True
Config['Show_Traders'] = False

Config['timing'] = \
{
    'tense': Tense.Present,
    'intervalSec': 1.0,
    'originSec': round(datetime(year=2015, month=1, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()), # for Tense.Past only
    'intervalsPerStep': BasicIntervalMin * 60
}

Config['structure'] = \
{
    'Name': 'SimpleEngine',
    'Traders':
    {
        'Trader1': 
        {
            'Type': 'SimpleTrader',
            'Name': '(1 - SimpleTrader',
            'Cared': True,
            'Strategies':
            {
                'Strategy1':
                {
                    'Type': 'SimpleStrategy',
                    'Name': '(1, 1 - SimpleStrategy)',
                    'Indicators':
                    {
                        'Indicator1':
                        {
                            'Type': 'SimpleIndicator',
                            'Name': '(1, 1, 1 - SimpleIndicator)',
                            'GStreams' :
                            {
                                'GStream.1.1.1.1':
                                {
                                    'Type': 'Binance',
                                    'Name': 'GSteram1 (1, 1, 1, 1)',
                                    'dataType': 'klines',
                                    'symbol':   'ALICEUSDT',
                                    'interval': Config['ginterval']
                                },
                                'GStream.2.1.1.1':
                                {
                                    'Type': 'Binance',
                                    'Name': 'GSteram1 (2, 1, 1, 1)',
                                    'dataType': 'klines',
                                    'symbol':   'DOTUSDT',
                                    'interval': Config['ginterval']
                                },
                            }
                        },
                        'Indicator2':
                        {
                            'Type': 'GoodIndicator',
                            'Name': '(2, 1, 1 - GoodIndicator)',
                            'GStreams' :
                            {
                                'GStream.1.2.1.1':
                                {
                                    'Type': 'Binance',
                                    'Name': 'GSteram1 (1, 2, 1, 1)',
                                    'dataType': 'klines',
                                    'symbol':   '1INCHUSDT',
                                    'interval': Config['ginterval']
                                },
                            }
                        },
                        'Indicator3':
                        {
                            'Type': 'ProductsPrices_Stream',
                            'Name': '(3, 1, 1 - Prices_Stream)',
                            'Quote': 'USDT',
                            'MaxStreams': 3333,
                            'Interval': Config['ginterval'],
                            'Live': True,
                            'GStreams' :
                            {
                            },
                        },
#                        'Indicator4':
#                        {
#                            'Type': 'ProductsPrices',
#                            'Name': '(4, 1, 1 - Prices)',
#                            'Quote': 'USDT',
#                            'StructuralParams_Price_Thread':
#                            {
#                                
#                            },
#                            'TimingParams_Price_Thread': 
#                            {
#                                'tense': Tense.Present,
#                               'intervalSec': 1.0,
#                                'originSec': 999,
#                                'intervalsPerStep': int(Config['binance_wait_sec']),
#                            },
#                            'Interval': Config['ginterval'],
#                            'GStreams' :
#                            {
#                            },
#                        },
                    },
                },
                'Strategy2':
                {
                    'Type': 'SimpleStrategy',
                    'Name': '(2, 1 - SimpleStrategy)',
                    'Indicators':
                    {
                        'Indicator1':
                        {
                            'Type': 'GoodIndicator',
                            'Name': '(1, 2, 1 - GoodIndicator)',
                            'GStreams' :
                            {
                                'GStream.1.1.2.1':
                                {
                                    'Type': 'Binance',
                                    'Name': 'GSteram1 (1, 1, 2, 1)',
                                    'dataType': 'klines',
                                    'symbol':   'BTCUSDT',
                                    'interval': Config['ginterval']
                                },
                                'GStream.2.1.2.1':
                                {
                                    'Type': 'Binance',
                                    'Name': 'GSteram1 (2, 1, 2, 1)',
                                    'dataType': 'klines',
                                    'symbol':   'ETHUSDT',
                                    'interval': Config['ginterval']
                                },
                            }
                        },
                        'Indicator2':
                        {
                            'Type': 'GoodIndicator',
                            'Name': '(2, 2, 1 - GoodIndicator)',
                            'GStreams' :
                            {
                                'GStream.1.2.2.1':
                                {
                                    'Type': 'Binance',
                                    'Name': 'GSteram3 (1, 2, 2, 1)',
                                    'dataType': 'klines',
                                    'symbol':   'MATICUSDT',
                                    'interval': Config['ginterval']
                                },
                            }
                        },
                    },
                },
            },
        },
    },
}


