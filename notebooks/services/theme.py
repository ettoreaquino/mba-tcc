import plotly.express as px

from itertools import cycle

def get_colors():

    color_scale = px.colors.sequential.Viridis[:-1]
    # color_scale = [
        # 'rgb(189,189,189)',
        # 'rgb(150,150,150)',
        # 'rgb(115,115,115)',
        # 'rgb(82,82,82)',
        # 'rgb(37,37,37)',
        # 'rgb(0,0,0)'
    # ]
    return cycle(list(set(color_scale)))