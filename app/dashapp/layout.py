from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import json
import plotly.express as px

def get_card(text1, id, type="primary"):
    card = html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                text1
                            ], className=f"text-xs font-weight-bold text-{type} text-uppercase mb-1"),
                            html.Div([
                                "00.00"
                            ], id = id, className="h5 mb-0 font-weight-bold text-gray-800")
                        ], className="col mr-2")
                    ], className="row no-gutters align-items-center")
                ], className="card-body")
            ], className=f"card border-left-{type} shadow h-100 py-2")
        ], className="col-xl-3 col-md-6 mb-4")
    return card

def get_dataTable(df):
    dataTable = html.Div([
        html.Div([
            html.Div([
                html.H6("Results Overview", className="m-0 font-weight-bold text-primary")
            ], className="card-header py-3 d-flex flex-row align-items-center justify-content-between"),
            html.Div([
                dash_table.DataTable(
                    id = 'main-table',
                    data = df.to_dict('records'), 
                    columns = [{"name": i, "id": i} for i in df.columns if i != 'id'],
                    fixed_rows={'headers': True},
                    sort_action='native',
                    style_table={'height': '400px', 'overflowY': 'auto'},
                )
            ], className="card-body")
        ], className="card shadow mb-4 h-100")
    ], className="col-xl-9 col-lg-8")
    return dataTable

def get_image(title, id):
    image = html.Div([
        html.Div([
            html.Div([
                html.H6(title, className="m-0 font-weight-bold text-primary")
            ], className="card-header py-3"),
            html.Div([
                html.Div([
                    
                ], className="pt-4", id=id),
            ], className="card-body p-0"),
        ], className="card shadow mb-4")
    ], className="col-xl-3 col-lg-4")
    return image

def get_scatter():
    image = html.Div([
        html.Div([
            html.Div([
                html.H6("Debris IoUs", className="m-0 font-weight-bold text-primary")
            ], className="card-header py-3"),
            html.Div([
                html.Div([

                ], id="scatter-plot-div"),
            ], className="card-body p-0"),
        ], className="card shadow mb-4 h-100")
    ], className="col-xl-3 col-lg-4")
    return image

def get_sidebar():
    sidebar = html.Ul([
        html.A([
            html.Div([
                html.I(className="fas fa-laugh-wink")
            ], className="sidebar-brand-icon rotate-n-15"),
            html.Div([
                "Glacier Mapping"
            ], className="sidebar-brand-text mx-3")
        ], href="#", className="sidebar-brand d-flex align-items-center justify-content-center"),
        html.Hr(className="sidebar-divider my-0"),
        html.Li([
            html.A([
                html.I(className="fas fa-fw fa-tachometer-alt"),
                html.Span("Dashboard")
            ], href="#", className="nav-link")
        ], className="nav-item active"),
        html.Hr(className="sidebar-divider"),
    ], className="navbar-nav bg-gradient-primary sidebar sidebar-dark accordion", id="accordionSidebar")
    return sidebar

def get_topbar(obj):
    topbar = html.Nav([
        html.Form([
            html.Span("Processed dirs: ", className="mr-2"),
            dcc.Dropdown(obj.get_all_processed_folders(), obj.get_processed_folder(), id='processed-dirs-dropdown', style={'width': '200px'}),
            html.Span("Runs: ", className="ml-2 mr-2"),
            dcc.Dropdown(obj.get_all_preds_folders(), obj.get_preds_folder(), id='runs-dropdown', style={'width': '200px'}),
        ], className="d-none d-sm-flex form-inline mr-auto ml-md-3 my-2 my-md-0 mw-100")
    ], className="navbar navbar-expand navbar-light bg-white topbar mb-4 static-top shadow")
    return topbar

def get_page_content(obj):
    page_content = html.Div([
        html.Div([
            html.H1("Dashboard", className="h3 mb-0 text-gray-800")
        ], className="d-sm-flex align-items-center justify-content-between mb-4"),
        html.Div([
            get_dataTable(obj.get_df()),
            get_scatter()
        ], className="row mb-4"),
        html.Div([
            get_image("Image", "x-div"),
            get_image("True Labels", "y-true-div"),
            get_image("Predicted Labels", "y-pred-div")
        ], className="row"),
        html.Div([
            get_card("Precision", "precision-div"),
            get_card("Recall", "recall-div", type="success"),
            get_card("IoU", "iou-div", type="info"),
            #get_card("AUC", "auc-div", type="warning"),
        ], className="row")
    ], className="container-fluid")
    return page_content

def get_layout(obj):
    layout = html.Div([
        get_sidebar(),
        html.Div([
            html.Div([
                get_topbar(obj),
                get_page_content(obj)
            ], id="content")
        ], className="d-flex flex-column", id="content-wrapper"),
    ], id='wrapper')
    return layout