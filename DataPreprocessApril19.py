import pandas as pd
import pandas_datareader as pdr

def preprocess_csv(file_path, volatility_window_size, return_window_size=60):
    # Read the CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Recreate timestamps as datetime objects and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')

    # Create Return column
    df['return'] = df['close'].pct_change()

    # Drop Open, High, Low, Close columns
    df = df.drop(columns=['open', 'high', 'low', 'close'])

    # Create past return columns
    for i in range(1, return_window_size + 1):
        df[f'return_{i}'] = df['return'].shift(i)

    # # Create economic series to include (treasury yields, inflation expectations, economic uncertainty)
    inflation = pdr.get_data_fred('T5YIE').shift(1)
    inflation.index = pd.to_datetime(inflation.index)
    df['inflation_expectation'] = inflation.reindex(df.index, method='ffill')

    two_year_treasury = pdr.get_data_fred('DGS2').shift(1)
    two_year_treasury.index = pd.to_datetime(two_year_treasury.index)
    df['two_year_treasury'] = two_year_treasury.reindex(df.index, method='ffill')

    ten_year_treasury = pdr.get_data_fred('DGS10').shift(1)
    ten_year_treasury.index = pd.to_datetime(ten_year_treasury.index)
    df['ten_year_treasury'] = ten_year_treasury.reindex(df.index, method='ffill')

    economic_uncertainty = pdr.get_data_fred('USEPUINDXD').shift(1)
    economic_uncertainty.index = pd.to_datetime(economic_uncertainty.index)
    df['economic_uncertainty'] = economic_uncertainty.reindex(df.index, method='ffill')

    # Create rolling volatility column, and lag it by the window size 
    df['volatility'] = df['return'].rolling(window=volatility_window_size).std()
    df['future_volatility'] = df['volatility'].shift(-volatility_window_size)

    # Drop rows with NaN values (the first row will have NaN in the return column)
    df = df.dropna()

    return df

if __name__ == "__main__":
    file_path = 'AAPL.csv'
    volatility_window_size = 60
    return_window_size = 60

    processed_df = preprocess_csv(file_path, volatility_window_size, return_window_size)

    print(processed_df.head(65))
    # Save the processed DataFrame to a new CSV file
    processed_df.to_csv('processed_AAPL.csv', index=False)