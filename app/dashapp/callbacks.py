from dash.dependencies import Output, Input
from dash import dcc
import plotly.express as px
import numpy as np

def register_callbacks(app, obj, layout):

    @app.callback(
        Output('x-div', 'children'),
        Input('main-table', 'active_cell'))
    def display_click_data(active_cell):
        if active_cell is None:
            row_id = 1
        else:
            row_id = active_cell['row_id']
        fname = str(obj.get_df().iloc[row_id]["tile_name"])
        fname = fname.replace("pred", "tiff")
        figpath = obj.get_processed_dir() / "test" / fname
        fig = np.load(figpath)[:,:,[5,4,2]]
        fig = (fig - np.min(fig, axis=(0,1)))/(np.max(fig, axis=(0,1)) - np.min(fig, axis=(0,1)))
        fig = px.imshow(fig)
        fig.update_layout(layout)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return dcc.Graph(figure=fig, id='x')
    
    @app.callback(
        Output('y-true-div', 'children'),
        Input('main-table', 'active_cell'))
    def display_click_data(active_cell):
        if active_cell is None:
            row_id = 1
        else:
            row_id = active_cell['row_id']
        fname = str(obj.get_df().iloc[row_id]["tile_name"])
        fname = fname.replace("pred", "tiff")
        figpath = obj.get_processed_dir() / "test" / fname
        x = np.load(figpath)
        mask = np.sum(x[:,:,:5], axis=2) == 0
        fname = fname.replace("tiff", "mask")
        figpath = obj.get_processed_dir() / "test" / fname
        y = np.load(figpath)
        _y = np.zeros((y.shape[0], y.shape[1], 3))
        for i in range(3):
            _y[:,:,i][y == i] = 1
        _y[mask] = 0
        y = _y
        fig = px.imshow(y)
        fig.update_layout(layout)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(dict(showscale=False, coloraxis=None), selector={'type':'heatmap'})
        return dcc.Graph(figure=fig, id='y-true')

    @app.callback(
        Output('y-pred-div', 'children'),
        Input('main-table', 'active_cell'))
    def display_click_data(active_cell):
        if active_cell is None:
            row_id = 1
        else:
            row_id = active_cell['row_id']
        fname = str(obj.get_df().iloc[row_id]["tile_name"])
        fname = fname.replace("pred", "tiff")
        figpath = obj.get_processed_dir() / "test" / fname
        x = np.load(figpath)
        mask = np.sum(x[:,:,:5], axis=2) == 0
        fname = fname.replace("tiff", "pred")
        figpath = obj.get_preds_dir() / fname
        y = np.load(figpath)
        y[mask] = 0
        fig = px.imshow(y)
        fig.update_layout(layout)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(dict(showscale=False, coloraxis=None), selector={'type':'heatmap'})
        return dcc.Graph(figure=fig, id='y-true')

    @app.callback(
        Output('precision-div', 'children'),
        Input('main-table', 'active_cell'))
    def display_click_data(active_cell):
        if active_cell is None:
            row_id = 1
        else:
            row_id = active_cell['row_id']
        ci_precision = str(np.round(obj.get_df().iloc[row_id]["ci_precision"]*100, 2))
        debris_precision = str(np.round(obj.get_df().iloc[row_id]["debris_precision"]*100, 2))
        return f"Clean Ice: {ci_precision}%, Debris: {debris_precision}%"

    @app.callback(
        Output('recall-div', 'children'),
        Input('main-table', 'active_cell'))
    def display_click_data(active_cell):
        if active_cell is None:
            row_id = 1
        else:
            row_id = active_cell['row_id']
        ci_recall = str(np.round(obj.get_df().iloc[row_id]["ci_recall"]*100, 2))
        debris_recall = str(np.round(obj.get_df().iloc[row_id]["debris_recall"]*100, 2))
        return f"Clean Ice: {ci_recall}%, Debris: {debris_recall}%"

    @app.callback(
        Output('iou-div', 'children'),
        Input('main-table', 'active_cell'))
    def display_click_data(active_cell):
        if active_cell is None:
            row_id = 1
        else:
            row_id = active_cell['row_id']
        ci_IoU = str(np.round(obj.get_df().iloc[row_id]["ci_IoU"]*100, 2))
        debris_IoU = str(np.round(obj.get_df().iloc[row_id]["debris_IoU"]*100, 2))
        return f"Clean Ice: {ci_IoU}%, Debris: {debris_IoU}%"

    @app.callback(
        Output('runs-dropdown', 'options'),
        Input('processed-dirs-dropdown', 'value'))
    def on_processed_dropdown_click(value):
        obj.set_processed_folder(value)
        return obj.get_all_preds_folders()

    
    @app.callback(
        Output('main-table', 'data'),
        Input('runs-dropdown', 'value'))
    def on_runs_dropdown_click(value):
        obj.set_preds_folder(value)
        df = obj.get_df()
        df = df.to_dict('records')
        return df

    @app.callback(
        Output('scatter-plot-div', 'children'),
        Input('runs-dropdown', 'value'))
    def on_runs_dropdown_click_scatter(value):
        obj.set_preds_folder(value)
        df = obj.get_df()
        fig = px.scatter(df, x="tile_name", y="debris_IoU")
        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_layout(yaxis_range=[0,1])
        
        return dcc.Graph(figure=fig, style={'height': '100%'})