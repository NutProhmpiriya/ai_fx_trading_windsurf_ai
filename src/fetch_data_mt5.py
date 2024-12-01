import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        mt5.shutdown()
        return False
    return True

def fetch_historical_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, year=2023):
    """
    Fetch historical data from MT5 for 2023-2024
    """
    if not initialize_mt5():
        return None
    
    # Set timezone to UTC
    timezone = pytz.timezone("Etc/UTC")
    
    # Define time period
    end_date = datetime(year, 12, 31, tzinfo=timezone)
    start_date = datetime(year, 1, 1, tzinfo=timezone)
    
    # Fetch the data
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None:
        print(f"Failed to fetch data for {symbol}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Save to CSV
    csv_filename = f"{symbol}_{timeframe}M_{year}_data.csv"
    df.to_csv('src/data/raw/' + csv_filename, index=False)
    print(f"Data saved to {csv_filename}")
    
    return df

def create_candlestick_chart(df, symbol, timeframe, year):
    """
    Create interactive candlestick chart using Plotly
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])

    # Update layout
    fig.update_layout(
        title=f'{symbol} Price Chart',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    # Save the chart
    html_filename = f"{symbol}_{timeframe}M_{year}_chart.html"
    fig.write_html('src/data/raw/' + html_filename)
    print(f"Chart saved to src/data/raw/{html_filename}")
    
    # Show the chart
    fig.show()

def main():
    # List of symbols to fetch
    symbols = [
        # "EURUSD", 
        # "GBPUSD", 
        "USDJPY"
    ]
    year = 2023
    timeframe = mt5.TIMEFRAME_M5  # 1-hour timeframe
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        df = fetch_historical_data(symbol, timeframe, year)
        if df is not None:
            create_candlestick_chart(df, symbol, timeframe, year)
    
    mt5.shutdown()

if __name__ == "__main__":
    main()