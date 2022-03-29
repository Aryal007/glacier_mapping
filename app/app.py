from dash import Dash
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os, json
from dashapp.layout import get_layout
from dashapp.callbacks import register_callbacks
from pathlib import Path

external_stylesheets = ['https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i']
app = Dash(__name__, external_stylesheets=external_stylesheets)

data_dir = Path("/data/baryal/HKH/processed_L07_2005")
preds_dir = data_dir / "preds"
test_files = os.listdir(data_dir / "test")
df = pd.read_csv(preds_dir / "metadata.csv", index_col=0)
df = df.sort_values('tile_name').reset_index()
df = df.round(4)
df = df.drop(["index"], axis=1)
df["id"] = df.index.tolist()
layout = go.Layout(
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=45, #bottom margin
        t=0, #top margin
    )
)
with open(preds_dir / 'conf.json') as f:
    lines = f.readlines()
model_parameters = json.loads(lines[0])

app.layout = get_layout(df, model_parameters)

register_callbacks(app, df, data_dir, preds_dir, layout)

if __name__ == '__main__':
    app.run_server(debug=True)
