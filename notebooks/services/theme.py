import plotly.express as px

from itertools import cycle

def get_colors():

    #color_scale = px.colors.qualitative.Plotly
    # color_scale = px.colors.qualitative.Safe
    # color_scale = px.colors.sequential.Viridis[:-1]

    # Color Blind 6
    # color_scale = [
    #     'rgb(215,48,39)',
    #     'rgb(252,141,89)',
    #     'rgb(254,224,144)',
    #     'rgb(224,243,248)',
    #     'rgb(145,191,219)',
    #     'rgb(69,117,180)'
    # ]
    color_scale = [
        'rgb(255,255,204)', # amarelo
        'rgb(199,233,180)',
        'rgb(127,205,187)',
        'rgb(65,182,196)',
        'rgb(29,145,192)',
        'rgb(34,94,168)',
        'rgb(12,44,132)'
    ]


    # Color blind 5
    #color_scale = [
    #    'rgb(255,255,204)',
    #    'rgb(161,218,180)',
    #    'rgb(65,182,196)',
    #    'rgb(44,127,184)',
    #    'rgb(37,52,148)']

    # Gray Scale
    # color_scale = [
    #     'rgb(189,189,189)',
    #     'rgb(150,150,150)',
    #     'rgb(115,115,115)',
    #     'rgb(82,82,82)',
    #     'rgb(37,37,37)',
    #     'rgb(0,0,0)'
    # ]
    return cycle(list(set(color_scale)))

def get_color_list():

    return [
        'rgb(199,233,180)',
        'rgb(127,205,187)',
        'rgb(65,182,196)',
        'rgb(29,145,192)',
        'rgb(34,94,168)',
        'rgb(12,44,132)'
    ]
