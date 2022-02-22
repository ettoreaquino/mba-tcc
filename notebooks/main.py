from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.api as sm

from plotly.subplots import make_subplots

from services import (translation, theme)

color_cycle = theme.get_colors()

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
                    line=dict(color=next(color_cycle)))

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
                    line=dict(color=next(color_cycle)))
        hourly = go.Scatter(
                    name='hour',
                    x=df.resample('h').mean().index,
                    y=df.resample('h').mean().speed,
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        day = go.Scatter(
                    name='day',
                    x=df.resample('d').mean().index,
                    y=df.resample('d').mean().speed,
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        month = go.Scatter(
                    name='month',
                    x=df.resample('m').mean().index,
                    y=df.resample('m').mean().speed,
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        year = go.Scatter(
                    name='year',
                    x=df.resample('y').mean().index,
                    y=df.resample('y').mean().speed,
                    mode='lines',
                    line=dict(color=next(color_cycle)))
        box_fig = px.box(df,labels={"value": "Wind Speed (m/s)","time": "Time"})
        box = box_fig['data'][0]


        fig = make_subplots(rows=5, cols=1,
                            vertical_spacing=0.025,
                            shared_yaxes=True)
        fig.update_layout(height=1000,
                          title_text="Wind Speed series and its averages over time")

        fig.add_trace(min10, row=1, col=1)
        fig.add_trace(hourly, row=2, col=1)
        fig.add_trace(day, row=3, col=1)
        fig.add_trace(month, row=4, col=1)
        fig.add_trace(year, row=5, col=1)

        fig.update_yaxes(title_text="Time Series", row=1, col=1)
        fig.update_yaxes(title_text="Hourly", row=2, col=1)
        fig.update_yaxes(title_text="Daily", row=3, col=1)
        fig.update_yaxes(title_text="Monthly", row=4, col=1)
        fig.update_yaxes(title_text="Yearly", row=5, col=1)
        
        fig.show()

    def decompose(self, period:str, model: str, plot:bool=True, overlay_trend:bool=False):
        switch = {
            'h': {'sample': 'h', 'period': 365*24, 'title': 'Hourly'},
            'd': {'sample': 'd', 'period': 365, 'title': 'Daily'},
            'w': {'sample': 'w', 'period': int(365/7), 'title': 'Weekly'},
            'm': {'sample': 'm', 'period': 12, 'title': 'Monthly'}
        }

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
                            title_text="{} Series decomposition".format(params['title']),
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

    def stationarity(self, period:str, verbose:bool=False, plot:bool=True):

        switch = {
            'h': {'sample': 'h','title': 'Hourly'},
            'd': {'sample': 'd','title': 'Daily'},
            'w': {'sample': 'w','title': 'Weekly'},
            'm': {'sample': 'm','title': 'Monthly'}
        }

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
                            title_text="Stationarity Analysis",
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