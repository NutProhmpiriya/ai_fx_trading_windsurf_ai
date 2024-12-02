import pandas as pd

def load_forex_data(file_path):
    """
    Load forex data from a CSV file and preprocess it.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert time column to datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Sort by time
    df = df.sort_index()
    
    return df
