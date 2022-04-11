from dash.dependencies import Output, Input
from dash import dcc
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
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

    @app.callback(
        Output('roc-div', 'children'),
        Input('main-table', 'active_cell'))
    def display_click_data(active_cell):
        classes = {0: "Background",
                1: "Clean Ice",
                2: "Debris"}
        if active_cell is None:
            row_id = 1
        else:
            row_id = active_cell['row_id']
        fname = str(obj.get_df().iloc[row_id]["tile_name"])
        fname = fname.replace("pred", "tiff")
        figpath = obj.get_processed_dir() / "test" / fname
        x = np.load(figpath)
        mask = np.sum(x[:,:,:5], axis=2) == 0
        true_fname = fname.replace("tiff", "mask")
        y_true = np.load(obj.get_processed_dir() / "test" / true_fname)
        pred_fname = fname.replace("tiff", "pred")
        figpath = obj.get_preds_dir() / pred_fname
        y_pred = np.load(figpath)
        y_true[mask], y_pred[mask] = 0, 0

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        for i in range(len(set(y_true.flatten()))):
            _y_true = (y_true[np.invert(mask)].flatten() == i).astype(np.int32)
            _y_score = y_pred[:,:,i][np.invert(mask)].flatten()

            fpr, tpr, _ = roc_curve(_y_true, _y_score)
            auc_score = roc_auc_score(_y_true, _y_score)

            name = f"{classes[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return dcc.Graph(figure=fig)