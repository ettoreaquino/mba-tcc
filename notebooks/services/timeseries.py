from datetime import timedelta

import pandas as pd
def get_resample_info(rule: str, interface: str) -> dict:

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

    sample_info = get_resample_info(rule)
    resampler = dataset.resample(rule_info['rule'], closed='left')

    df = resampler.mean()
    df.rename(columns={df.columns[0]:'mean'}, inplace=True)
    df['std'] = resampler.std()
    df['min'] = resampler.min()
    df['max'] = resampler.max()
    df['median'] = resampler.median()
    
    return df

def missing_stats(original_df: pd.DataFrame, missing_df:pd.DataFrame, interface: dict, verbose:bool=False) -> pd.DataFrame:
    
    missing_l = [r.Index for r in missing_df.itertuples()]
    d = {}
    aux = 0
    for i,time in enumerate(missing_l):

        if i == 0:
            d.update({aux: {'values': [time]}})
            continue

        lapse = timedelta(minutes=10)
        delta = time - missing_l[i-1]

        if delta == lapse:
            d[aux]['values'].append(time)
        else:
            d[aux]['begin'] = d[aux]['values'][0]
            d[aux]['end'] = d[aux]['values'][-1]
            d[aux]['delta'] = d[aux]['values'][-1] - d[aux]['values'][0]
            d[aux]['missing'] = len(d[aux]['values'])

            aux += 1
            d.update({aux: {'values': [time]}})

    missing_df = pd.DataFrame([{
        'missing': d[b].get('missing'),
        'begin': d[b].get('begin'),
        'end': d[b].get('end'),
        'delta': d[b].get('delta')
    } for b in d][:-1])

    length = original_df.shape[0]
    n_missing = missing_df.shape[0]
    percentage = round(missing_df.shape[0]/original_df.shape[0]*100,4)
    longest = missing_df.sort_values(['missing'], ascending=False).iloc[0]
    frequent = missing_df.groupby('missing')\
            .count()\
            .sort_values('begin', ascending=False)\
            .reset_index()\
            .iloc[0]

    if verbose:
        report = interface['report'].format(length=length,
                                            missing=n_missing,
                                            percentage=percentage,
                                            missing_sequence=longest.missing,
                                            delta=longest.delta,
                                            begin=longest.begin,
                                            end=longest.end,
                                            frequent_missing = frequent.missing,
                                            frequent_count = frequent.begin)
        print(report)
    
    return missing_df.sort_values(by=['missing'], ascending=[False])