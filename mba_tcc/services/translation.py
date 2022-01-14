import pandas as pd

def undisclosed(csv_path:str) -> pd.DataFrame:
    '''Builds a Pandas.DataFrame from a csv.
    '''
    df = pd.read_csv(csv_path,
                     usecols = ['TIME','WIND_SPD_TOP'],
                     index_col=['TIME'],
                     parse_dates=['TIME']).asfreq('10T')
    df.rename(columns = {'WIND_SPD_TOP':'speed'}, inplace = True)
    df.index.rename('time', inplace=True)

    return df