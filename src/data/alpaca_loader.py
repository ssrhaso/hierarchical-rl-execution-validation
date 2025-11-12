"""
LOAD REAL MARKET DATA FROM ALPACA API

- 1 minute OHLCV (open, high, low, close, volume) bar data for specified tickers and date range
- US Equities (9300 AM - 1600  EST trading hours)
"""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)



class AlpacaDataLoader:
    """Professional Alpaca market data fetcher, with error handling"""
        
    TIMEFRAMES = {
        '1Min': TimeFrame.Minute,
        '5Min': TimeFrame(5, "Min") ,   
        '15Min': TimeFrame(15, "Min"),
        '1Hour': TimeFrame.Hour,
        '1Day': TimeFrame.Day
    }
    
    def __init__(self):
        try:
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if not (api_key and secret_key):
                raise RuntimeError("MUST supply ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment or .env file.")

            self.client = StockHistoricalDataClient(api_key, secret_key)
            logger.info(f"Alpaca client initialized (authenticated)")
        except Exception as e:
            logger.error(f"Failed to initialise AlpacaDataLoader: {e}")
            raise


    # DOWNLOAD BARS
    def download_bars(
        self,
        symbol: str = "TSLA",
        start_date : str = "2025-11-10",
        end_date : str = "2025-11-10",
        timeframe: str = "1Min", 
    ) -> Optional[pd.DataFrame]: 
        
        """Download historical OHLCV (open, high, low, close, volume) bar data for a given symbol and date range"""

        # VALIDATE
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Valid options are: {list(self.TIMEFRAMES.keys())}"
            )
        
        # REQUEST
        try:
            
            # CREATE REQUEST
            request = StockBarsRequest(
                symbol_or_symbols = [symbol],
                timeframe = self.TIMEFRAMES[timeframe],
                start = datetime.strptime(start_date, "%Y-%m-%d"),
                end = datetime.strptime(end_date, "%Y-%m-%d"),
                limit = 10000
            )
            
            # FETCH DATA
            logger.info(f"Requesting {timeframe} bars for {symbol} from {start_date} to {end_date}.")
            bars = self.client.get_stock_bars(request)
            df = bars.df

            
            # CLEAN and ALWAYS RESET INDEX
            df = df.reset_index()
            print("After reset_index:")
            print(df.columns.tolist())

            # Rename columns only if 9 columns present
            if df.shape[1] == 9:
                df.columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
            else:
                raise RuntimeError(f"Unexpected DataFrame shape after reset_index: {df.shape}")

            # KEEP ONLY OHLCV + VWAP
            df = df[["timestamp", "open", "high", "low", "close", "volume", "vwap"]]
            df.columns = df.columns.str.lower()


            
            # VALIDATE QUALITY OF DATA
            if df.isnull().any().any():
                logger.warning(f"Found {df.isnull().sum().sum()} missing values in downloaded data for {symbol}.")
            
            logger.info(f"Successfully downloaded {len(df)} bars for {symbol}.")
            return df
        
        except Exception as e:
            logger.error(f"Error downloading bars for {symbol}: {e}")
            raise RuntimeError(f"Failed to download bars for {symbol}") from e
     
        
    def download_multiple(
        self, symbols: list, start_date: str, end_date: str, timeframe: str = "1Min"
    ) -> dict:
        """Download Data for multiple symbols"""
        
        data = {}
        for symbol in symbols:
            logger.info(f"Downloading data for {symbol}...")
            data[symbol] = self.download_bars(symbol, start_date, end_date, timeframe)
        
        logger.info(f"Completed downloading data for {len(symbols)} symbols.")
        return data
    
    
    
    