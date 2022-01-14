from datetime import datetime
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

    def check_missing(self):
        missing = self.data.loc[(self.data.isnull().speed == True)]
        fmt = '''{} missing values found!\nRepresenting {}% of the dataset'''
        print(fmt.format(missing.shape[0], round(missing.shape[0]/self.data.shape[0]*100,4)))

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