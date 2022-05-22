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

def _load_interface(language: str):
    if language not in ['english', 'portuguese']:
        raise ValueError('Invalid language {}'.format(language))

    with open("services/{}.yml".format(language)) as file:
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
    
    def __init__(self, name: str, csv_path: str, language: str='english'):
        
        self.name = name
        self.data = translation.undisclosed(csv_path=csv_path)
        self.__interface = _load_interface(language=language)

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
        df = self.data.copy()

        if export:
            title_text = None
        else:
            title_text = self.__interface['series']['title']

        fig = make_subplots(rows=8, cols=1,
                            vertical_spacing=0.025,
                            shared_yaxes=True,
                            shared_xaxes=True)
        fig.update_layout(height=1000,
                          title_text=title_text)

        min10 = go.Scatter(
                    name='10 min',
                    x=df.index,
                    y=df.speed,
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        fig.add_trace(min10, row=1, col=1)
        fig.update_yaxes(title_text="10 min", row=1, col=1)

        for i, sample in enumerate(self.__interface['sampling']):
            
            series = timeseries.resample(dataset=df, rule=sample['rule'])
            trace = go.Scatter(name=sample['name'],
                               x=series.index,
                               y=series['mean'],
                               mode='lines',
                               line=dict(color=next(color_cycle)))
            fig.add_trace(trace, row=i+2, col=1)
            fig.update_yaxes(title_text=sample['title'], row=i+2, col=1)
        
        fig.show()

    def decompose(self, period:str, model: str, diff: int=0, plot:bool=True, overlay_trend:bool=False, export:bool=False):
        
        df = self.data.copy()
        sample = list(filter(lambda x: x['rule'] == period, self.__interface['sampling']))[0]

        if export:
            title_text = None
        else:
            title_text = self.__interface['decomposition']['title'].format(sample['title'])

        if diff > 0:
            df = df.diff(diff)

        series = timeseries.resample(dataset=df, rule=sample['rule'])[['mean']].dropna(subset=['mean'])
        decomposition = sm.tsa.seasonal_decompose(series, period=eval(sample['period']), model=model)
        
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residue = decomposition.resid
        
        if plot:
            series = go.Scatter(
                        name='series',
                        x=series.index,
                        y=series['mean'],
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
                                vertical_spacing=0.025,
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

            fig.update_yaxes(title_text=self.__interface['decomposition']['yaxis']['title'], row=1, col=1)
            fig.update_yaxes(title_text=self.__interface['decomposition']['yaxis']['trend'], row=2, col=1)
            fig.update_yaxes(title_text=self.__interface['decomposition']['yaxis']['season'], row=3, col=1)
            fig.update_yaxes(title_text=self.__interface['decomposition']['yaxis']['residue'], row=4, col=1)

            if overlay_trend:
                trend = go.Scatter(
                        name='trend',
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        mode='lines',
                        opacity=0.8,
                        line=dict(color=next(color_cycle)))
                fig.add_trace(trend, row=1, col=1)

            fig.show()

    def stationarity(self, period:str, diff: int=0, verbose:bool=False, plot:bool=True, export:bool=False):
        df = self.data.copy()
        switch = {
            'h': {'sample': 'h','title': 'Hourly'},
            'd': {'sample': 'd','title': 'Daily'},
            'w': {'sample': 'w','title': 'Weekly'},
            'm': {'sample': 'm','title': 'Monthly'}
        }

        if export:
            title_text = None
        else:
            title_text = "Análise de Estacionariedade"

        params = switch.get(period)

        if diff > 0:
            df = df.diff(diff)

        series = df.resample(params['sample']).mean().dropna(subset=['speed'])
        test = sm.tsa.stattools.adfuller(series, autolag='AIC')

        self.adf = test[0]
        self.p_value=test[1]
        self.usedlag = test[2]
        self.nobs = test[3]
        self.rejected_ho = True if self.adf < test[4]['5%'] else False
        self.acf_array = sm.tsa.stattools.acf(series.dropna(), alpha=0.05)
        self.pacf_array = sm.tsa.stattools.pacf(series.dropna(), alpha=0.05)
        
        stationarity = 'Série Temporal ESTACIONÁRIA' if self.rejected_ho else 'Série Temporal NÃO-ESTACIONÁRIA' 


        if verbose:
            print('=========================================================')
            print('{:^57s}'.format('Augmented Dickey-Fuller Test'))
            print('{:^57s}'.format(params['title'].upper()))
            print('---------------------------------------------------------')
            print('Estatística ADF:')
            print('{adf}'.format(adf=self.adf))
            print('---------------------------------------------------------')
            print('p-valor:')
            print('{p_value}'.format(p_value=self.p_value))
            print('---------------------------------------------------------')
            print('lags utilizados:')
            print('{usedlag}'.format(usedlag=self.usedlag))
            print('---------------------------------------------------------')
            print('N observações utilizadas:')
            print('{nobs}'.format(nobs=self.nobs))
            print('=========================================================')
            print('Valores Críticos:')
            for k, v in test[4].items():
                print('\t{}: {}'.format(k,v))
            print('---------------------------------------------------------')
            print('Hipótese Nula Rejeitada? - {rejected_ho}'.format(rejected_ho = self.rejected_ho))
            print(stationarity)
            print('=========================================================')

        if plot:
            color_cycle = theme.get_colors()

            series_trace = go.Scatter(name='series',
                                      x=series.index,
                                      y=series.speed,
                                      mode='lines',
                                      line=dict(color=next(color_cycle)))

            acf_traces = _build_traces(corr_array=self.acf_array)
            pacf_traces = _build_traces(corr_array=self.pacf_array)

            fig = make_subplots(rows=2, cols=2,
                                vertical_spacing=0.075,
                                specs=[[{"colspan": 2}, None],
                                       [{},{}]],
                                subplot_titles=("{} Série<br>Dickey-Fuller p-valor: {}".format(params['title'], round(self.p_value,4)),
                                                "Autocorrelação",
                                                "Autocorrelação Parcial"))

            fig.update_layout(height=1000,
                            title_text=title_text,
                            xaxis_showticklabels=True,
                            showlegend=False)

            fig.update_yaxes(zerolinecolor='#000000')

            fig.add_trace(series_trace, row=1, col=1)
            for t in acf_traces: fig.add_trace(t, row=2, col=1)
            for t in pacf_traces: fig.add_trace(t, row=2, col=2)
            fig.show()

    def build_sets(self, period:str, split: str, diff: int=0, plot: bool=False, export:bool=False):
        '''
        Returns a dataset, a trainingset and a testset based on a period
        and a split refererence. 
        '''
        
        df = self.data.copy()
        sample_info = list(filter(lambda x: x['rule'] == period, self.__interface['sampling']))[0]

        if diff > 0:
            df = df.diff(diff)

        self.dataset = timeseries.resample(dataset=df, rule=period).interpolate(method='linear')

        if export:
            title_text = None
        else:
            title_text = self.__interface['train']['title']

        splittime = datetime.strptime(split, sample_info['idx_fmt'])
        self.trainset = self.dataset.loc[(self.dataset.index < splittime)]
        self.testset  = self.dataset.loc[(self.dataset.index >= splittime)]

        if plot:
            train = go.Scatter(
                        name=self.__interface['train']['train'],
                        x=self.trainset.index,
                        y=self.trainset['mean'],
                        mode='lines',
                        line=dict(color=next(color_cycle)))

            #Appends last element of training to connect line
            testset_plot = pd.concat([self.trainset.iloc[-1:], self.testset])
            test = go.Scatter(
                        name=self.__interface['train']['test'],
                        x=testset_plot.index,
                        y=testset_plot['mean'],
                        mode='lines',
                        line=dict(color=next(color_cycle)))

            fig = go.Figure()
            fig.add_trace(train)
            fig.add_trace(test)
            fig.update_layout(title_text=title_text,
                                xaxis_showticklabels=True)

            fig.show()

    def reindex_series(self):
        '''Void function to reindex time series between begin and end'''
        min_time = min(self.data.index)
        max_time = max(self.data.index)

        idx = pd.period_range(min_time, max_time, freq='10T')
        df = self.data.reindex(idx)
        
        self.data = df
        print('Dados da torre reindexados entre {} e {}'.format(min_time, max_time))

    def save(self):

        with open('{}.pkl'.format(self.name), 'wb') as file:
            pickle.dump(self, file)

        print('File {}.pkl, saved.'.format(self.name))

