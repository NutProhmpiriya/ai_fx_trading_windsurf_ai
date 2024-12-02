import pandas as pd

def load_forex_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess forex data from CSV file
    
    Args:
        data_path (str): Path to CSV file containing forex data
        
    Returns:
        pd.DataFrame: Preprocessed forex data
    """
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df
