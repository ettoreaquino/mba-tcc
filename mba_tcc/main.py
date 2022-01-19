from datetime import datetime, timedelta
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

from plotly.subplots import make_subplots

from services import translation

viridis_scale = px.colors.sequential.Viridis

from services import (translation)

class WindSpeedTower():
    
    def __init__(self, csv_path: str):
        
        self.data = translation.undisclosed(csv_path='datasets/wind.csv')

    def stats_missing(self, verbose:bool=False) -> pd.DataFrame:
        missing = self.data.loc[(self.data.isnull().speed == True)]
        
        missing_l = [r.Index for r in missing.itertuples()]
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

        length = self.data.shape[0]
        n_missing = missing.shape[0]
        percentage = round(missing.shape[0]/self.data.shape[0]*100,4)
        longest = missing_df.sort_values(['missing'], ascending=False).iloc[0]
        frequent = missing_df.groupby('missing')\
                .count()\
                .sort_values('begin', ascending=False)\
                .reset_index()\
                .iloc[0]

        fmt = '''
    =========================================================
    Length of Time Series:
    {length}
    ---------------------------------------------------------
    Number of Missing Values:
    {missing}
    ---------------------------------------------------------
    Percentage of Missing Values:
    {percentage} %
    =========================================================
    Stats for Gaps

    Longest Gap (series of consecutive missing):
    {missing_sequence} missing in a row for a total of {delta}
    Between {begin} and {end}
    ---------------------------------------------------------
    Most frequent gap size (series of consecutive NA series):
    {frequent_missing} missing in a row (occurring {frequent_count} times)
    =========================================================
        '''.format(
            length=length,
            missing=n_missing,
            percentage=percentage,
            missing_sequence=longest.missing,
            delta=longest.delta,
            begin=longest.begin,
            end=longest.end,
            frequent_missing = frequent.missing,
            frequent_count = frequent.begin)
        
        if verbose:
            print(fmt)
        
        return missing_df

    def get_range(self, begin: datetime, end: datetime) -> pd.DataFrame:
        '''Returns the dataset between the selected range'''
        mask = (self.data.index > begin) & (self.data.index <= end)

        return self.data.loc[mask]
        
    def plot_date(self, year: int, month: int=None, day: int=None):
        
        if year == None:
            raise ValueError('Year cannot be empty')

        if year!= None and month != None and day != None:
            df = self.data.loc[(self.data.index.year == year) & (self.data.index.month == month) & (self.data.index.day == day)]
            name = "{}-{}-{}".format(year, month, day)
        elif year!= None and month != None and day == None:
            df = self.data.loc[(self.data.index.year == year) & (self.data.index.month == month)]
            name = "{}-{}".format(year, month)
        elif year!= None and month == None and day == None:
            df = self.data.loc[(self.data.index.year == year)]
            name = "{}".format(year)

        year = go.Scatter(
                    name=name,
                    x=df.index,
                    y=df.speed,
                    mode='lines',
                    line=dict(color=viridis_scale[0]))

        box_fig = px.box(df, labels={"value": "Wind Speed (m/s)","time": "Time"})
        box = box_fig['data'][0]


        fig = go.Figure()
        fig.update_layout(height=300)
        fig.add_trace(year)

        fig.show()

        
    def plot_series(self):
        '''Plot linegraphs to the Time Series in different time scales
        '''
        df = self.data.copy()
        
        min10 = go.Scatter(
                    name='10 min',
                    x=df.index,
                    y=df.speed,
                    mode='lines',
                    line=dict(color=viridis_scale[0]))
        hourly = go.Scatter(
                    name='hour',
                    x=df.resample('h').mean().index,
                    y=df.resample('h').mean().speed,
                    mode='lines',
                    line=dict(color=viridis_scale[2]))
        day = go.Scatter(
                    name='day',
                    x=df.resample('d').mean().index,
                    y=df.resample('d').mean().speed,
                    mode='lines',
                    line=dict(color=viridis_scale[4]))
        month = go.Scatter(
                    name='month',
                    x=df.resample('m').mean().index,
                    y=df.resample('m').mean().speed,
                    mode='lines',
                    line=dict(color=viridis_scale[6]))
        year = go.Scatter(
                    name='year',
                    x=df.resample('y').mean().index,
                    y=df.resample('y').mean().speed,
                    mode='lines',
                    line=dict(color=viridis_scale[8]))
        box_fig = px.box(df,labels={"value": "Wind Speed (m/s)","time": "Time"})
        box = box_fig['data'][0]


        fig = make_subplots(rows=5, cols=1, subplot_titles=("10 min", "Hourly", "Daily", "Monthly", "Yearly"), shared_yaxes=True)
        fig.update_layout(height=1500)
        fig.add_trace(min10, row=1, col=1)
        fig.add_trace(hourly, row=2, col=1)
        fig.add_trace(day, row=3, col=1)
        fig.add_trace(month, row=4, col=1)
        fig.add_trace(year, row=5, col=1)
        
        fig.show()

    def reindex_series(self):
        '''Void function to reindex time series between begin and end'''
        min_time = min(self.data.index)
        max_time = max(self.data.index)

        idx = pd.period_range(min_time, max_time, freq='10T')
        df = self.data.reindex(idx)
        
        self.data = df
        print('Tower data reindexed between {} and {}'.format(min_time, max_time))