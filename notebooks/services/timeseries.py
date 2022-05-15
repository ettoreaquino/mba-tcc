import pandas as pd

def get_resample_info(rule: str) -> dict:

    sample_switch = {
        'h':  {'rule': 'h',  'period': 365*24,     'title': 'Hourly',    'idx_fmt': '%Y-%m-%dT%H:%M:%S'},
        'd':  {'rule': 'd',  'period': 365,        'title': 'Daily',     'idx_fmt': '%Y-%m-%d'},
        'w':  {'rule': 'w',  'period': int(365/7), 'title': 'Weekly',    'idx_fmt': '%Y-%m-%d'},
        'm':  {'rule': 'm',  'period': 12,         'title': 'Monthly',   'idx_fmt': '%Y-%m-%d'},
        'q':  {'rule': 'q',  'period': 4,          'title': 'Quarterly', 'idx_fmt': '%Y-%m-%d'},
        '2q': {'rule': '2q', 'period': 2,          'title': 'Bi-Annual', 'idx_fmt': '%Y-%m-%d'},
        'y':  {'rule': 'y', 'period': 1,           'title': 'Yearly',    'idx_fmt': '%Y-%m-%d'},
    }
    
    return sample_switch.get(rule)

def resample(dataset: pd.DataFrame, rule: str):
    '''
    Resamples the original series based on a resampling rule:
     - h: hourly
     - d: daily
     - w: weekly
     - m: monthly
     - q: quarterly
     - 2q: bi-annual
     
    The resampled dataset indicates all of the statistics regarding the resampled time window.
    '''

    rule_info = get_resample_info(rule)
    resampler = dataset.resample(rule_info['rule'], closed='left')

    df = resampler.mean()
    df.rename(columns={df.columns[0]:'mean'}, inplace=True)
    df['std'] = resampler.std()
    df['min'] = resampler.min()
    df['max'] = resampler.max()
    df['median'] = resampler.median()
    
    return df
