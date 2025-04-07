# Libraries 
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import talib
import mplfinance as mpf

script_started = datetime.datetime.now()

print(f'Work started at {datetime.datetime.now()}')

# Basic Folder Set-up
base_path = 'd:\\Projects\\Stock_Market\\data'
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
save_plot_path = os.path.join(base_path, f'selected\\plots_{today_str}')
save_csv_path = os.path.join(base_path, f'selected\\')

for path in [save_plot_path, save_csv_path]:
    os.makedirs(path, exist_ok=True)

# Set up end and start times for data download (last 180 days)
end_date = datetime.datetime.now()
start_date = datetime.datetime.now() - datetime.timedelta(days=180)

# List of selected tickers for Swedish stocks
tickers_sweden = ['HEM.ST', 'INDT.ST', 'BERG-B.ST']
tickers_sweden = sorted(tickers_sweden)
# omxs30 = ['SWED-A.ST', 'KINV-B.ST', 'VOLV-B.ST', 'NIBE-B.ST', 'AZN.ST', 'SEB-A.ST', 'ERIC-B.ST', 'SINCH.ST', 'SCA-B.ST', 'ELUX-B.ST', 'SKF-B.ST', 'ESSITY-B.ST', 'GETI-B.ST', 'SHB-A.ST', 'EVO.ST', 'SAND.ST', 'NDA-SE.ST', 'ALFA.ST', 'TEL2-B.ST', 'HM-B.ST', 'INVE-B.ST', 'ASSA-B.ST', 'HEXA-B.ST', 'ABB.ST', 'ATCO-B.ST', 'SAAB-B.ST', 'ATCO-A.ST', 'TELIA.ST', 'BOL.ST', 'SBB-B.ST']


# Function to download stock data
def download_stock_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, interval='1d')
        data[ticker] = df
    
    return data

# Function to calculate technical indicators
def add_technical_indicators(df):    

    # Flatten multi-level columns, if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # Keep only 'Price' level

    # Ensure OHLCV columns are float64 for TA-Lib compatibility
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    # Moving Averages
    df['SMA_05'] = talib.SMA(df['Close'].values, timeperiod=5)
    df['SMA_08'] = talib.SMA(df['Close'].values, timeperiod=8)
    df['SMA_13'] = talib.SMA(df['Close'].values, timeperiod=13)
    df['SMA_21'] = talib.SMA(df['Close'].values, timeperiod=21)
    df['EMA_50'] = talib.EMA(df['Close'].values, timeperiod=50)
    df['EMA_200'] = talib.EMA(df['Close'].values, timeperiod=200)
    df['VolumeSMA_14'] = talib.SMA(df['Volume'].values, timeperiod=14)

    # Momentum Indicators
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'].values)
    df['BB_Up'], df['BB_Mid'], df['BB_Low'] = talib.BBANDS(df['Close'].values, timeperiod=20)
    df['Plus_DMI'] = talib.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values)
    df['Minus_DMI'] = talib.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values)
    df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values)
    df['SAR'] = talib.SAR(df['High'].values, df['Low'].values, acceleration=0.02, maximum=0.2)
    df['SAR_MoreSensitive'] = talib.SAR(df['High'].values, df['Low'].values, acceleration=0.04, maximum=0.4)
    df['SAR_LessSensitive'] = talib.SAR(df['High'].values, df['Low'].values, acceleration=0.01, maximum=0.1)

    # Ichimoku
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    df['Chikou_Span'] = df['Close'].shift(-26)

    # Volatility
    df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    df['NATR'] = talib.NATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    df['TRANGE'] = talib.TRANGE(df['High'].values, df['Low'].values, df['Close'].values)

    # Other momentum & oscillators
    df['APO'] = talib.APO(df['Close'].values)
    df['CCI'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    df['CMO'] = talib.CMO(df['Close'].values, timeperiod=14)
    df['MFI'] = talib.MFI(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values, timeperiod=14)
    df['MOM'] = talib.MOM(df['Close'].values, timeperiod=10)
    df['ROC'] = talib.ROC(df['Close'].values, timeperiod=10)
    df['ROCP'] = talib.ROCP(df['Close'].values, timeperiod=10)
    df['ROCR'] = talib.ROCR(df['Close'].values, timeperiod=10)
    df['ROCR100'] = talib.ROCR100(df['Close'].values, timeperiod=10)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
    df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(df['High'].values, df['Low'].values, df['Close'].values)
    df['STOCHRSI_K'], df['STOCHRSI_D'] = talib.STOCHRSI(df['Close'].values)
    df['TRIX'] = talib.TRIX(df['Close'].values, timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(df['High'].values, df['Low'].values, df['Close'].values)
    df['WILLR'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)

    # Volume indicators
    df['OBV'] = talib.OBV(df['Close'].values, df['Volume'].values)
    df['AD'] = talib.AD(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values)

    # CMF (Chaikin Money Flow)
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
          (df['High'] - df['Low']) * df['Volume']
    df['CMF'] = mfv.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # VPT
    df['VPT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()

    # BOP
    df['BOP'] = talib.BOP(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)


# Function to generate and save plots SET 1
def plot_stock_data_1(ticker, df):
    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [3, 3, 2, 2, 3]}, figsize=(10, 10), sharex=True)

    # Plot SMA and EMA
    axs[0].plot(df['Close'], color='black', label='Close')
    axs[0].plot(df['SMA_05'], linestyle='-.', color='green', label='SMA05')
    axs[0].plot(df['SMA_08'], linestyle='-.', color='red', label='SMA08')
    axs[0].plot(df['SMA_13'], linestyle='-.', color='blue', label='SMA13')
    axs[0].plot(df['EMA_50'], linestyle='dotted', color='grey', label='EMA50')
    axs[0].set_title(f'{ticker} - Price with MA', loc='left')
    axs[0].legend(loc="upper left")

    # Plot Bollinger Bands
    axs[1].plot(df['Close'], color='black', label='Close')
    # axs[1].plot(df[['BB_Up', 'BB_Mid', 'BB_Low']], label=['BB_Up', 'BB_Mid', 'BB_Low'])
    axs[1].plot(df['BB_Up'], linestyle='--', color='blue', label='BB_Up')
    axs[1].plot(df['BB_Mid'], linestyle='-', color='orange', label = 'BB_Mid')
    axs[1].plot(df['BB_Low'], linestyle='--', color='red', label = 'BB_Low')
    axs[1].set_title(f'{ticker} - Price with BB', loc='left')
    axs[1].legend(loc="upper left")

    # Plot MACD
    axs[2].plot(df['MACD'], label='MACD')
    axs[2].plot(df['MACD_Signal'], '--', color='orange', label='MACD_Signal')
    c = ['red' if v < 0 else 'green' for v in df['MACD_Hist']]
    axs[2].bar(df.index, df['MACD_Hist'], color=c)
    axs[2].set_title(f'{ticker} - MACD', loc='left')
    axs[2].legend(loc="upper left")
    axs[2].set_ylim([min(df['MACD'].min(), df['MACD_Signal'].min(), df['MACD_Hist'].min()), max(df['MACD'].max(), df['MACD_Signal'].max(), df['MACD_Hist'].max())])
    axs[2].set_yticks(np.arange(df['MACD'].min(), df['MACD'].max(), (df['MACD'].max() - df['MACD'].min()) / 5), minor=True)

    # Plot RSI
    axs[3].plot(df['RSI'], color='orange', label='RSI')
    axs[3].plot(df['MFI'], color='blue', label='MFI', linestyle='--')
    axs[3].set_title(f'{ticker} - RSI', loc='left')
    axs[3].legend(loc="upper left")
    axs[3].axhline(y=70, color='r', linestyle='--')
    axs[3].axhline(y=50, color='lightgrey', linestyle='--')
    axs[3].axhline(y=30, color='g', linestyle='--')
    
    # Plot ADX and DMI
    axs[4].plot(df['ADX'], color='black', label='ADX')
    axs[4].plot(df['Plus_DMI'], color='blue', label='Plus_DMI')
    axs[4].plot(df['Minus_DMI'], color='red', label='Minus_DMI')
    axs[4].axhline(y=30, color='lightgrey', linestyle='--')
    axs[4].axhline(y=25, color='grey', linestyle='--')
    axs[4].axhline(y=20, color='lightgrey', linestyle='--')
    axs[4].set_title(f'{ticker} - ADX and DMI', loc='left')
    axs[4].legend(loc="upper left")

    # Format x-axis
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlim([datetime.datetime.now() - datetime.timedelta(days=120), datetime.datetime.now()])
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right', which='both')

    # Set title and layout
    fig.suptitle(ticker)
    plt.tight_layout()

    # Save plot    
    os.makedirs(save_plot_path, exist_ok=True)
    plt.savefig(os.path.join(save_plot_path, f'{ticker}_plot_1_{datetime.datetime.now().strftime("%Y-%m-%d")}.png'), dpi=300)
    plt.show()

# Function to generate and save plots SET 2
def plot_stock_data_2(ticker, df):
    fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 3, 3, 3]}, figsize=(10, 10), sharex=True)

    # Additional Indicators (Grouped by Type)
    # Plot SMA and EMA
    axs[0].plot(df['Close'], color='black', label='Close')
    axs[0].plot(df['SMA_05'], linestyle='-.', color='green', label='SMA05')
    axs[0].plot(df['SMA_08'], linestyle='-.', color='red', label='SMA08')
    axs[0].plot(df['SMA_13'], linestyle='-.', color='blue', label='SMA13')
    axs[0].plot(df['EMA_50'], linestyle='dotted', color='grey', label='EMA50')
    axs[0].set_title(f'{ticker} - Price with MA', loc='left')
    axs[0].legend(loc="upper left")
    
    # CCI
    axs[1].axhline(y=100, color='lightgrey', linestyle='--')
    axs[1].axhline(y=0, color='lightgrey', linestyle='--')
    axs[1].plot(df.index, df['CCI'], label='CCI', color='blue')
    axs[1].axhline(y=-100, color='lightgrey', linestyle='--')
    axs[1].set_title(f'{ticker} - CCI', loc='left')
    axs[1].legend(loc="upper left")
      
    # Parabolic SAR
    axs[2].plot(df.index, df['Close'], label='Close', color='black')
    axs[2].plot(df.index, df['SAR'], label='SAR', color='blue', linestyle='--')
    axs[2].plot(df.index, df['SAR_MoreSensitive'], label='SAR_MoreSensitive', color='red', linestyle='--')
    axs[2].plot(df.index, df['SAR_LessSensitive'], label='SAR_LessSensitive', color='green', linestyle='dotted')
    axs[2].set_title(f'{ticker} Parabolic SAR', loc='left')
    axs[2].legend(loc="upper left")
    
    # Stochastic Oscillators
    axs[3].plot(df.index, df['STOCH_K'], label='%K (Main)', color='blue')
    axs[3].plot(df.index, df['STOCH_D'], label='%D (Signal)', color='red', linestyle='--')
    # axs[3].plot(df.index, df['STOCHF_K'], label='Fast %K', color='green')
    # axs[3].plot(df.index, df['STOCHF_D'], label='Fast %D', color='orange')
    # axs[3].plot(df.index, df['STOCHRSI_K'], label='StochRSI %K', color='purple')
    # axs[3].plot(df.index, df['STOCHRSI_D'], label='StochRSI %D', color='brown')
    axs[3].set_title(f'{ticker} - Stochastic Oscillators', loc='left')
    axs[3].legend(loc="upper left")

    # Format x-axis
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlim([datetime.datetime.now() - datetime.timedelta(days=120), datetime.datetime.now()])
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right', which='both')

    # Set title and layout
    fig.suptitle(ticker)
    plt.tight_layout()

    # Save plot    
    os.makedirs(save_plot_path, exist_ok=True)
    plt.savefig(os.path.join(save_plot_path, f'{ticker}_plot_2_{datetime.datetime.now().strftime("%Y-%m-%d")}.png'), dpi=300)
    plt.show()

# Function to generate and save plots SET 3
def plot_stock_data_3(ticker, df):
    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3]}, figsize=(10, 10), sharex=True)

    # Additional Indicators (Grouped by Type)

    # Plot SMA and EMA
    axs[0].plot(df['Close'], color='black', label='Close')
    axs[0].plot(df['SMA_05'], linestyle='-.', color='green', label='SMA05')
    axs[0].plot(df['SMA_08'], linestyle='-.', color='red', label='SMA08')
    axs[0].plot(df['SMA_13'], linestyle='-.', color='blue', label='SMA13')
    axs[0].plot(df['EMA_50'], linestyle='dotted', color='grey', label='EMA50')
    axs[0].set_title(f'{ticker} - Price with MA', loc='left')
    axs[0].legend(loc="upper left")
    
    # True Range Indicators
    axs[1].plot(df.index, df['NATR'], label='NATR', color='blue')
    axs[1].plot(df.index, df['TRANGE'], label='TRANGE', color='red')
    axs[1].set_title(f'{ticker} - True Range Indicators', loc='left')
    axs[1].legend(loc="upper left")  

    # Rate of Change
    axs[2].plot(df.index, df['ROC'], label='ROC', color='blue')
    axs[2].plot(df.index, df['ROCP'], label='ROCP', color='red', linestyle='--')
    axs[2].plot(df.index, df['ROCR'], label='ROCR', color='green', linestyle='dotted')
    # axs[2].plot(df.index, df['ROCR100'], label='ROCR100', color='orange')
    axs[2].set_title(f'{ticker} - Rate of Change', loc='left')
    axs[2].legend(loc="upper left")
    
    # BOP Balance of power 
    axs[3].plot(df.index, df['BOP'], label='BOP', color='blue')
    axs[3].axhline(y=0, color='lightgrey', linestyle='--')
    axs[3].set_title(f'{ticker} - BOP Balance of Power', loc='left')
    axs[3].legend(loc="upper left")
    
    # Oscillators
    axs[4].plot(df.index, df['APO'], label='APO', color='blue')
    axs[4].plot(df.index, df['CMO'], label='CMO', color='red')
    axs[4].plot(df.index, df['MOM'], label='MOM', color='green')
    axs[4].plot(df.index, df['TRIX'], label='TRIX', color='orange')
    axs[4].plot(df.index, df['ULTOSC'], label='ULTOSC', color='purple')
    axs[4].plot(df.index, df['WILLR'], label='WILLR', color='brown')
    axs[4].set_title(f'{ticker} - Oscillators', loc='left')
    axs[4].legend(loc="upper left")    

    # Format x-axis
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlim([datetime.datetime.now() - datetime.timedelta(days=120), datetime.datetime.now()])
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right', which='both')

    # Set title and layout
    fig.suptitle(ticker)
    plt.tight_layout()

    # Save plot    
    os.makedirs(save_plot_path, exist_ok=True)
    plt.savefig(os.path.join(save_plot_path, f'{ticker}_plot_3_{datetime.datetime.now().strftime("%Y-%m-%d")}.png'), dpi=300)
    plt.show()

# Function to generate and save plots SET 4
def plot_stock_data_4(ticker, df):
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 3]}, figsize=(10, 10), sharex=True)
    
    axs[0].plot(df['Close'], color='black', label='Close')
    axs[0].plot(df['SMA_05'], linestyle='-.', color='green', label='SMA05')
    axs[0].plot(df['SMA_08'], linestyle='-.', color='red', label='SMA08')
    axs[0].plot(df['SMA_13'], linestyle='-.', color='blue', label='SMA13')
    axs[0].plot(df['EMA_50'], linestyle='dotted', color='grey', label='EMA50')
    axs[0].set_title(f'{ticker} - Price with MA', loc='left')
    axs[0].legend(loc="upper left")
    
    axs[1].plot(df['Close'], color='black', label='Close Price')
    axs[1].plot(df['Tenkan_sen'], label='Tenkan-sen (Conversion Line)', linestyle='--')
    axs[1].plot(df['Kijun_sen'], label='Kijun-sen (Base Line)', linestyle='--')
    axs[1].plot(df['Senkou_Span_A'], label='Senkou Span A', linestyle='--')
    axs[1].plot(df['Senkou_Span_B'], label='Senkou Span B', linestyle='--')
    axs[1].fill_between(df.index, df['Senkou_Span_A'], df['Senkou_Span_B'], where=df['Senkou_Span_A'] >= df['Senkou_Span_B'], color='lightgreen', alpha=0.3)
    axs[1].fill_between(df.index, df['Senkou_Span_A'], df['Senkou_Span_B'], where=df['Senkou_Span_A'] < df['Senkou_Span_B'], color='lightcoral', alpha=0.3)
    axs[1].plot(df['Chikou_Span'], label='Chikou Span', linestyle='--')
    axs[1].set_title(f'{ticker} Ichimoku Cloud', loc ='left')
    axs[1].legend(loc='upper left')

    # Format x-axis
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlim([datetime.datetime.now() - datetime.timedelta(days=120), datetime.datetime.now()])
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right', which='both')

    # Set title and layout
    fig.suptitle(ticker)
    plt.tight_layout()

    # Save plot    
    os.makedirs(save_plot_path, exist_ok=True)
    plt.savefig(os.path.join(save_plot_path, f'{ticker}_plot_4_{datetime.datetime.now().strftime("%Y-%m-%d")}.png'), dpi=300)
    plt.show()        

# Function to generate and save plots SET 5

def plot_stock_data_5(ticker, df):
    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [2, 2, 2, 2, 2]}, figsize=(10, 10), sharex=True)
    
    axs[0].plot(df['Close'], color='black', label='Close')
    axs[0].plot(df['SMA_05'], linestyle='-.', color='green', label='SMA05')
    axs[0].plot(df['SMA_08'], linestyle='-.', color='red', label='SMA08')
    axs[0].plot(df['SMA_13'], linestyle='-.', color='blue', label='SMA13')
    axs[0].plot(df['EMA_50'], linestyle='dotted', color='grey', label='EMA50')
    axs[0].set_title(f'{ticker} - Price with MA', loc='left')
    axs[0].legend(loc="upper left")
    
    # Volume
    axs[1].bar(df.index, df['Volume'], label='Volume', color='grey')
    axs[1].plot(df['VolumeSMA_14'], color='blue', label='VolumeSMA_14')
    axs[1].set_title(f'{ticker} - Volume', loc='left')
    axs[1].legend(loc="upper left")
    axs[1].set_ylim([0, df['Volume'].max()])
    axs[1].set_yticks(np.arange(0, df['Volume'].max(), df['Volume'].max() / 10), minor=True)
    
    # OBV
    axs[2].plot(df.index, df['OBV'], label='OBV', color='blue', linestyle='-.')
    axs[2].plot(df.index, df['AD'], label='A/D Line', color='red')
    axs[2].axhline(y=0, color='lightgrey', linestyle='--')
    axs[2].set_title(f'{ticker} On-Balance Volume (OBV) and Accumulation/Distribution (A/D) Line', loc = 'left')
    axs[2].legend(loc='upper left')
    
    # CMF
    axs[3].plot(df.index, df['CMF'], label='CMF', color='blue')
    axs[3].axhline(y=0, color='lightgrey', linestyle='--')
    axs[3].set_title(f'{ticker} Chaikin Money Flow (CMF)', loc = 'left')
    axs[3].legend(loc='upper left')
    
    # VPT
    axs[4].plot(df.index, df['VPT'], label='VPT', color='purple')
    axs[4].axhline(y=0, color='lightgrey', linestyle='--')
    axs[4].set_title(f'{ticker} Volume Price Trend (VPT)', loc = 'left')
    axs[4].legend(loc='upper left')

    # Format x-axis
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlim([datetime.datetime.now() - datetime.timedelta(days=120), datetime.datetime.now()])
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right', which='both')

    # Set title and layout
    fig.suptitle(ticker)
    plt.tight_layout()

    # Save plot
    
    os.makedirs(save_plot_path, exist_ok=True)
    plt.savefig(os.path.join(save_plot_path, f'{ticker}_plot_5_volume_{datetime.datetime.now().strftime("%Y-%m-%d")}.png'), dpi=300)
    plt.show()  

# Function to plot stock data with candlestick patterns and formatted x-axis
# Function to identify candlestick patterns
def identify_patterns(df):
    patterns = {
        'Hammer': talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close']),
        'Shooting Star': talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close']),
        'Doji': talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close']),
        'Engulfing': talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close']),
        'Morning Star': talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close']),
        'Evening Star': talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    }
    pattern_positions = {k: [] for k in patterns}
    for pattern, data in patterns.items():
        pattern_positions[pattern] = data[data != 0].index
    return pattern_positions

def plot_stock_data_6(ticker, df):
     # Configure the plot with specific x-axis formatting
    fig, ax = mpf.plot(
        df[-60:],
        type='candle',
        style='classic',
        title=ticker,
        returnfig=True, 
        datetime_format='%Y-%m-%d', 
        xrotation=45,
        figscale = 1,
        volume=True
    )

    ax = ax[0]  # Get the main axis
    # Identify candlestick patterns
    patterns = identify_patterns(df)
    
    # Annotate patterns
    for pattern, positions in patterns.items():
        for pos in positions:
            ax.annotate(pattern, (mdates.date2num(pos), df.loc[pos, 'Close']),
                        xytext=(mdates.date2num(pos), df.loc[pos, 'Close'] + df['Close'].std()),
                        arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    
    # Set title and layout
    fig.suptitle(ticker)
    # plt.tight_layout()

    # Save plot
    os.makedirs(save_plot_path, exist_ok=True)
    plt.savefig(os.path.join(save_plot_path, f'{ticker}_plot_6_candles_{datetime.datetime.now().strftime("%Y-%m-%d")}.png'), dpi=300)
    plt.show() 
        
# Main workflow
def main():
    # Download stock data
    stock_data = download_stock_data(tickers_sweden, start_date, end_date)

    # Calculate technical indicators for each stock
    for ticker, df in stock_data.items():
        add_technical_indicators(df)

    # Plot Set 1
    for ticker, df in stock_data.items():
        plot_stock_data_1(ticker, df)

    # Plot Set 2
    for ticker, df in stock_data.items():
        plot_stock_data_2(ticker, df)

    # Plot Set 3
    for ticker, df in stock_data.items():
        plot_stock_data_3(ticker, df)
    
    # Plot Set 4
    for ticker, df in stock_data.items():
        plot_stock_data_4(ticker, df)
    
    # Plot Set 5
    for ticker, df in stock_data.items():
        plot_stock_data_5(ticker, df)
           
    # Plot 6 candlestick patterns
    for ticker, df in stock_data.items():
        plot_stock_data_6(ticker, df)
        
    # Combine and export stock data for further analysis
    combined_data = pd.concat(stock_data, axis=0, keys=stock_data.keys())
    # combined_data.to_csv(f'stock_data_combined_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv')
    
    os.makedirs(save_plot_path, exist_ok=True)
    combined_data.to_csv(os.path.join(save_plot_path, f'stock_data_combined_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'))
    selected_stocks = combined_data.reset_index().groupby(['level_0']).tail(1)
    selected_stocks.to_csv(os.path.join(save_plot_path, f'stock_data_selected_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'))
    selected_stocks = selected_stocks.T.reset_index()
    selected_stocks.columns = selected_stocks.iloc[0]
    selected_stocks.to_csv(os.path.join(save_plot_path, f'stock_data_selected_transposed_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'))
    print(f'Work started at: {script_started}')
    print(f'Work done at: {datetime.datetime.now()}')
    print(f'Time it took to run the script: {datetime.datetime.now() - script_started}') 

if __name__ == "__main__":
    main()

