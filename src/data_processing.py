import pandas as pd
import numpy as np

def load_and_format_temperature(temp_path='data/temperaturapulpos.csv'):
    """Loads and formats the raw temperature data from CSV."""
    df = pd.read_csv(temp_path, sep=';', decimal=",")
    df['Fecha'] = pd.to_datetime(df['Ano'].astype(str) + 
                                 df['Mes'].astype(str).str.zfill(2) + 
                                 df['Dia'].astype(str).str.zfill(2))
    df.drop(['Ano', 'Mes', 'Dia', 'Hora'], axis=1, inplace=True)
    
    # Calculate daily means and set date as index
    df = df.groupby('Fecha').mean()
    
    # Filter data from 2012 onward
    df = df[df.index >= '2012-01-01']
    return df

def load_and_format_laying_data(laying_path='data/puestaspulpos.xlsx'):
    """Loads and formats the octopus egg-laying data from Excel."""
    df2 = pd.read_excel(laying_path)
    df2 = df2.iloc[:366]  # Ensure consistent day range (handles leap years)
    df2.drop(columns=[col for col in df2.columns if 'Unnamed' in str(col)], inplace=True)
    df2.fillna(0, inplace=True)
    
    # Melt dataframe to long format
    df_melted = df2.melt(id_vars='Fecha', value_vars=[col for col in df2.columns if isinstance(col, int)],
                         var_name='Year', value_name='Puestas')
                         
    # Combine day/month with the correct year
    df_melted['Date'] = df_melted.apply(
        lambda row: row['Fecha'].replace(year=row['Year']), axis=1
    )
    df_melted.set_index('Date', inplace=True)
    df_melted.drop(['Fecha', 'Year'], axis=1, inplace=True)
    
    # Binarize the target variable (any laying event is 1)
    df_melted[df_melted != 0] = 1
    return df_melted

def create_final_dataframe(temp_path, laying_path):
    """Creates the final merged, cleaned, and resampled dataframe."""
    df_temp = load_and_format_temperature(temp_path)
    df_laying = load_and_format_laying_data(laying_path)
    
    # Merge the two dataframes
    df_merged = pd.merge(df_temp, df_laying, left_index=True, right_index=True, how='inner')
    
    # Resample to weekly means/max
    weekly_df = df_merged.resample('W').agg({'Temp': 'mean', 'Puestas': 'max'})
    weekly_df.dropna(inplace=True)
    
    return weekly_df

def create_lag_features(df, lags=[2, 4, 6, 8]):
    """Creates lagged temperature features."""
    for i in lags:
        df[f'Temp_Lag_{i}w'] = df['Temp'].shift(i)
    
    df.dropna(inplace=True)
    df.drop('Temp', axis=1, inplace=True)
    return df
