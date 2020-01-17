import numpy as numpy
import pandas as pd
import glob as glob

def process_df(path, ticker):
    '''
    @param path, string, contains csv with stock data
    '''
    
    # Read data to DataFrame
    df = pd.read_csv(path)
    df.rename(columns = lambda x: x.strip(), inplace = True)
    
    
    # Initialize result
    result = pd.DataFrame()
    
    # Sort by date
    result['Date'] = pd.to_datetime(df['Date'])
    result[ticker + '_Volume'] = df['Volume'].astype(float)
    result[ticker + '_Close'] = df['Close/Last'].apply(lambda x: x.split('$')[1]).astype(float)
    result[ticker + '_Open'] = df['Open'].apply(lambda x: x.split('$')[1]).astype(float)
    result[ticker + '_High'] = df['High'].apply(lambda x: x.split('$')[1]).astype(float)
    result[ticker + '_Low'] = df['Low'].apply(lambda x: x.split('$')[1]).astype(float)
    
    # Return data in chronological order
    return result.sort_values(by = "Date", ascending = True)

def process_directory(prefix, suffix):
    
    # Files in directory
    files = glob.glob(prefix + '*' + suffix)
    
    # Populate DataFrame with data from each stock in directory
    for i, f in enumerate(files):
        
        # Ticker symbol
        ticker = f[len(prefix):-len(suffix)]
        
        # Initialize result
        if i == 0:
            data = process_df(f, ticker)
            
        # Add to result
        else:
            data = data.merge(process_df(f, ticker), 
                              how = "inner",
                              on = "Date")
            
    return data