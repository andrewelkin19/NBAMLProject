import pandas as pd
import numpy as np

# Load contract data from CSV to Pandas dataframe.
def loadContractData(filename='contracts.csv'):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"The file: {filename} could not be found.")
        return None

def convertDollar(val):
    if isinstance(val, str) and '$' in val:
        # Remove the dollar sign and commas, then convert to int
        return int(val.replace('$', '').replace(',', ''))
    return val

# Clean loaded data for easier reading
def cleanContractData(df):
    # Replace Monetary Values With Numbers
    df = df.fillna(0)
    df[df.columns[1:]] = df[df.columns[1:]].applymap(convertDollar)
    return df # Return the cleaned DataFrame

def removeExtraHeaderRows(df):
    #Convert the 'Rk' cols to numeric; non-numeric values become NaN
    df['Rk'] = pd.to_numeric(df['Rk'], errors='coerce')
    # Keep only valid numeric values
    df = df.dropna(subset=['Rk'])
    # Convert 'Rk' to int
    df['Rk'] = df['Rk'].astype(int)
    return df


if __name__ == '__main__':
    df = loadContractData()
    if df is not None:
        df = cleanContractData(df)
        df = removeExtraHeaderRows(df)
    print(df)
        
    
