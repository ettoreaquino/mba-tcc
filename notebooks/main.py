from datetime import datetime, timedelta
from re import I
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.api as sm
import yaml

from plotly.subplots import make_subplots

from services import (translation, theme, timeseries)

color_cycle = theme.get_colors()

def load_tower(pickefile: str):

    with open(pickefile, 'rb') as file:
        Tower = pickle.load(file)

    return Tower

def _load_interface(interface: str):
    with open("services/portuguese.yml") as file:
        interface = yaml.load(file, Loader=yaml.SafeLoader)

    return interface

def _build_traces(corr_array):
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
    
    traces = []
    for x in range(len(corr_array[0])):
        
        traces.append(go.Scatter(x=(x,x),
                                 y=(0,corr_array[0][x]),
                                 mode='lines',
                                 line_color='#3f3f3f'))
    traces.append(go.Scatter(x=np.arange(len(corr_array[0])),
                             y=corr_array[0],
                             mode='markers',
                             marker_color='#1f77b4',
                             marker_size=10))
    traces.append(go.Scatter(x=np.arange(len(corr_array[0])),
                             y=upper_y,
                             mode='lines',
                             line_color='rgba(255,255,255,0)'))
    traces.append(go.Scatter(x=np.arange(len(corr_array[0])),
                             y=lower_y, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty',
                             line_color='rgba(255,255,255,0)'))
    return traces

class WindSpeedTower():
    
    def __init__(self, name: str, csv_path: str, interface: str='english'):
        
        self.name = name
        self.data = translation.undisclosed(csv_path=csv_path)
        self.__interface = _load_interface(interface=interface)

    def missing_stats(self, verbose:bool=False) -> pd.DataFrame:
        missing = self.data.loc[(self.data.isnull().speed == True)]
        
        return timeseries.missing_stats(original_df=self.data,
                                        missing_df=missing,
                                        interface=self.__interface['missing'],
                                        verbose=verbose)
        

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
                    line=dict(color=next(color_cycle)))

        box_fig = px.box(df, labels={"value": "Wind Speed (m/s)","time": "Time"})
        box = box_fig['data'][0]


        fig = go.Figure()
        fig.update_layout(height=300)
        fig.add_trace(year)

        fig.show()

        
    def plot_series(self, export:bool=False):
        '''Plot linegraphs to the Time Series in different time scales
        '''

        if export:
            title_text = None
        else:
            if self.__portuguese:
                title_text = "Série temporal de velocidade de vento e médias por período"
            else:
                title_text = "Wind Speed series and its averages over different time periods"

        df = self.data.copy()

        hourly = timeseries.resample(dataset=df, rule='h')
        daily = timeseries.resample(dataset=df, rule='d')
        weekly = timeseries.resample(dataset=df, rule='w')
        monthly = timeseries.resample(dataset=df, rule='m')
        quarterly = timeseries.resample(dataset=df, rule='q')
        half_annualy = timeseries.resample(dataset=df, rule='2q')
        yearly = timeseries.resample(dataset=df, rule='y')
        
        min10 = go.Scatter(
                    name='10 min',
                    x=df.index,
                    y=df.speed,
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        hourly = go.Scatter(
                    name='hour' if not self.__portuguese else 'hora',
                    x=hourly.index,
                    y=hourly['mean'],
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        day = go.Scatter(
                    name='day' if not self.__portuguese else 'dia',
                    x=daily.index,
                    y=daily['mean'],
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        week = go.Scatter(
                    name='week' if not self.__portuguese else 'semana',
                    x=weekly.index,
                    y=weekly['mean'],
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        month = go.Scatter(
                    name='month' if not self.__portuguese else 'mes',
                    x=monthly.index,
                    y=monthly['mean'],
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        quarter = go.Scatter(
                    name='quarter' if not self.__portuguese else 'trimestre',
                    x=quarterly.index,
                    y=quarterly['mean'],
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        half_annual = go.Scatter(
                    name='half-annual' if not self.__portuguese else 'semestre',
                    x=half_annualy.index,
                    y=half_annualy['mean'],
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        year = go.Scatter(
                    name='year' if not self.__portuguese else 'ano',
                    x=yearly.index,
                    y=yearly['mean'],
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        box_fig = px.box(df,labels={"value": "Wind Speed (m/s)","time": "Time"})
        box = box_fig['data'][0]


        fig = make_subplots(rows=8, cols=1,
                            vertical_spacing=0.025,
                            shared_yaxes=True,
                            shared_xaxes=True)
        fig.update_layout(height=1000,
                          title_text=title_text)

        fig.add_trace(min10, row=1, col=1)
        fig.add_trace(hourly, row=2, col=1)
        fig.add_trace(day, row=3, col=1)
        fig.add_trace(week, row=4, col=1)
        fig.add_trace(month, row=5, col=1)
        fig.add_trace(quarter, row=6, col=1)
        fig.add_trace(half_annual, row=7, col=1)
        fig.add_trace(year, row=8, col=1)

        fig.update_yaxes(title_text="Time Series", row=1, col=1)
        fig.update_yaxes(title_text="Hourly", row=2, col=1)
        fig.update_yaxes(title_text="Daily", row=3, col=1)
        fig.update_yaxes(title_text="Weekly", row=4, col=1)
        fig.update_yaxes(title_text="Monthly", row=5, col=1)
        fig.update_yaxes(title_text="Quarterly", row=6, col=1)
        fig.update_yaxes(title_text="Half-Annualy", row=7, col=1)
        fig.update_yaxes(title_text="Yearly", row=8, col=1)
        
        fig.show()

    def decompose(self, period:str, model: str, plot:bool=True, overlay_trend:bool=False, export:bool=False):
        switch = {
            'h': {'sample': 'h', 'period': 365*24, 'title': 'Hourly'},
            'd': {'sample': 'd', 'period': 365, 'title': 'Daily'},
            'w': {'sample': 'w', 'period': int(365/7), 'title': 'Weekly'},
            'm': {'sample': 'm', 'period': 12, 'title': 'Monthly'}
        }

        if export:
            title_text = None
        else:
            title_text = "{} Series decomposition".format(params['title'])

        params = switch.get(period)

        series = self.data.copy().resample(params['sample']).mean().dropna(subset=['speed'])
        decomposition = sm.tsa.seasonal_decompose(series, period=params['period'], model=model)
        
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residue = decomposition.resid
        
        if plot:
            series = go.Scatter(
                        name='series',
                        x=series.index,
                        y=series.speed,
                        mode='lines',
                        line=dict(color=next(color_cycle)))

            trend = go.Scatter(
                        name='trend',
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        mode='lines',
                        line=dict(color=next(color_cycle)))
            seasonal = go.Scatter(
                        name='seasonal',
                        x=decomposition.seasonal.index,
                        y=decomposition.seasonal,
                        mode='lines',
                        line=dict(color=next(color_cycle)))
            resid = go.Scatter(
                        name='resid',
                        x=decomposition.resid.index,
                        y=decomposition.resid,
                        mode='lines',
                        line=dict(color=next(color_cycle)))

            fig = make_subplots(rows=5, cols=1,
                                vertical_spacing=0.015,
                                shared_yaxes=True,
                                shared_xaxes=True)

            fig.update_layout(height=1000,
                            title_text=title_text,
                            xaxis4_showticklabels=True,
                            showlegend=False)

            fig.add_trace(series, row=1, col=1)
            fig.add_trace(trend, row=2, col=1)
            fig.add_trace(seasonal, row=3, col=1)
            fig.add_trace(resid, row=4, col=1)

            fig.update_yaxes(title_text="Time Series", row=1, col=1)
            fig.update_yaxes(title_text="Trend", row=2, col=1)
            fig.update_yaxes(title_text="Seasonal", row=3, col=1)
            fig.update_yaxes(title_text="Residue", row=4, col=1)

            if overlay_trend:
                trend = go.Scatter(
                        name='trend',
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        mode='lines',
                        opacity=0.5,
                        line=dict(color=next(color_cycle)))
                fig.add_trace(trend, row=1, col=1)

            fig.show()

    def stationarity(self, period:str, verbose:bool=False, plot:bool=True, export:bool=False):

        switch = {
            'h': {'sample': 'h','title': 'Hourly'},
            'd': {'sample': 'd','title': 'Daily'},
            'w': {'sample': 'w','title': 'Weekly'},
            'm': {'sample': 'm','title': 'Monthly'}
        }

        if export:
            title_text = None
        else:
            title_text = "Stationarity Analysis"

        params = switch.get(period)

        series = self.data.copy().resample(params['sample']).mean().dropna(subset=['speed'])
        test = sm.tsa.stattools.adfuller(series, autolag='AIC')

        self.adf = test[0]
        self.p_value=test[1]
        self.rejected_ho = True if self.adf < test[4]['5%'] else False
        stationarity = 'Time Series is STATIONARY' if self.rejected_ho else 'Time Series is NON-STATIONARY' 


        if verbose:
            print('=========================================================')
            print('{:^57s}'.format('Augmented Dickey-Fuller Test'))
            print('{:^57s}'.format(params['title'].upper()))
            print('---------------------------------------------------------')
            print('ADF Statistic:')
            print('{adf}'.format(adf=self.adf))
            print('---------------------------------------------------------')
            print('p-value:')
            print('{p_value}'.format(p_value=self.p_value))
            print('=========================================================')
            print('Critical Values:')
            for k, v in test[4].items():
                print('\t{}: {}'.format(k,v))
            print('---------------------------------------------------------')
            print('Rejected Null Hypothesis? - {rejected_ho}'.format(rejected_ho = self.rejected_ho))
            print(stationarity)
            print('=========================================================')

        if plot:
            color_cycle = theme.get_colors()

            acf_array = sm.tsa.stattools.acf(series.dropna(), alpha=0.05)
            pacf_array = sm.tsa.stattools.pacf(series.dropna(), alpha=0.05)

            series_trace = go.Scatter(name='series',
                                      x=series.index,
                                      y=series.speed,
                                      mode='lines',
                                      line=dict(color=next(color_cycle)))

            acf_traces = _build_traces(corr_array=acf_array)
            pacf_traces = _build_traces(corr_array=pacf_array)


            fig = make_subplots(rows=2, cols=2,
                                vertical_spacing=0.075,
                                specs=[[{"colspan": 2}, None],
                                       [{},{}]],
                                subplot_titles=("{} Series<br>Dickey-Fuller p-value: {}".format(params['title'], round(self.p_value,4)),
                                                "Autocorrelation",
                                                "Partial Autocorrelation"))

            fig.update_layout(height=1000,
                            title_text=title_text,
                            xaxis_showticklabels=True,
                            showlegend=False)

            fig.update_yaxes(zerolinecolor='#000000')

            fig.add_trace(series_trace, row=1, col=1)
            for t in acf_traces: fig.add_trace(t, row=2, col=1)
            for t in pacf_traces: fig.add_trace(t, row=2, col=2)
            fig.show()


    def reindex_series(self):
        '''Void function to reindex time series between begin and end'''
        min_time = min(self.data.index)
        max_time = max(self.data.index)

        idx = pd.period_range(min_time, max_time, freq='10T')
        df = self.data.reindex(idx)
        
        self.data = df
        print('Tower data reindexed between {} and {}'.format(min_time, max_time))

    def save(self):

        with open(self.name, 'wb') as file:
            pickle.dump(self, file)

        print('File {}, saved.'.format(self.name))

