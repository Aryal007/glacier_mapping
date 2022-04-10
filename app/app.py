from dash import Dash
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os, json
from dashapp.layout import get_layout
from dashapp.callbacks import register_callbacks
from dashapp import base
from pathlib import Path

external_stylesheets = ['https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i']
app = Dash(__name__, external_stylesheets=external_stylesheets)
layout = go.Layout(
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=45, #bottom margin
        t=0, #top margin
    )
)

obj = base.Base()
obj.set_df()

app.layout = get_layout(obj)

register_callbacks(app, obj, layout)

if __name__ == '__main__':
    app.run_server(debug=True)
