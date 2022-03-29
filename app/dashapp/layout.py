from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import json

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
                    style_table={'height': '280px', 'overflowY': 'auto'},
                )
            ], className="card-body")
        ], className="card shadow mb-4")
    ], className="col-xl-9 col-lg-8")
    return dataTable

def get_model_parameters(model_parameters):
    image = html.Div([
        html.Div([
            html.Div([
                html.H6("Model Parameters", className="m-0 font-weight-bold text-primary")
            ], className="card-header py-3"),
            html.Div([
                html.Div([
                    html.P([
                        f"Batch Size: {model_parameters['batch_size']}",
                        html.Br(),
                        f"Epochs: {model_parameters['epochs']}",
                        html.Br(),
                        f"Loss: {model_parameters['loss_opts']['name']}",
                        html.Br(),
                        f"Learning rate: {model_parameters['optim_opts']['args']['lr']}",
                        html.Br(),
                        f"Dropout: {model_parameters['model_opts']['args']['dropout']}",
                        html.Br(),
                        f"Input Channels: {model_parameters['model_opts']['args']['inchannels']}",
                        html.Br(),
                        f"Depth: {model_parameters['model_opts']['args']['net_depth']}",
                        html.Br(),
                        f"Output Channels: {model_parameters['model_opts']['args']['outchannels']}",
                        html.Br(),
                        f"Normalization: {model_parameters['normalize']}",
                    ])
                ], className="chart-pie pt-4"),
            ], className="card-body"),
        ], className="card shadow mb-4")
    ], className="col-xl-3 col-lg-4")
    return image

def get_image(title, id):
    image = html.Div([
        html.Div([
            html.Div([
                html.H6(title, className="m-0 font-weight-bold text-primary")
            ], className="card-header py-3"),
            html.Div([
                html.Div([
                    
                ], className="chart-pie pt-4", id=id),
            ], className="card-body"),
        ], className="card shadow mb-4")
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

def get_topbar():
    topbar = html.Nav([
        html.Form([
            html.Div([
                dcc.Input(type="text", className="form-control bg-light border-0 small", placeholder="Search for..."),
                html.Div([
                    html.Button([
                        html.I(className="fas fa-search fa-sm")
                    ], className="btn btn-primary", type="button")
                ], className="input-group-append")
            ], className="input-group")
        ], className="d-none d-sm-inline-block form-inline mr-auto ml-md-3 my-2 my-md-0 mw-100 navbar-search")
    ], className="navbar navbar-expand navbar-light bg-white topbar mb-4 static-top shadow")
    return topbar

def get_page_content(df, model_parameters):
    page_content = html.Div([
        html.Div([
            html.H1("Dashboard", className="h3 mb-0 text-gray-800")
        ], className="d-sm-flex align-items-center justify-content-between mb-4"),
        html.Div([
            get_dataTable(df),
            get_model_parameters(model_parameters)
        ], className="row"),
        html.Div([
            get_image("Image", "x-div"),
            get_image("True Labels", "y-true-div"),
            get_image("Predicted Labels", "y-pred-div")
        ], className="row"),
        html.Div([
            get_card("Precision", "precision-div"),
            get_card("Recall", "recall-div", type="success"),
            get_card("IoU", "iou-div", type="info"),
            get_card("AUC", "auc-div", type="warning"),
        ], className="row")
    ], className="container-fluid")
    return page_content

def get_layout(df, model_parameters):
    layout = html.Div([
        get_sidebar(),
        html.Div([
            html.Div([
                get_topbar(),
                get_page_content(df, model_parameters)
            ], id="content")
        ], className="d-flex flex-column", id="content-wrapper"),
    ], id='wrapper')
    return layout

def get_layout_old(df):
    layout = html.Div([
    html.Div([
        html.Header([
            html.Div([
                html.Div([
                    html.Div([
                        html.A([
                            html.Span("Interactive Visualization")
                        ], href="/", className="d-flex align-items-center mb-3 mb-lg-0 me-lg-auto text-white text-decoration-none"),
                        #html.Div("Debris and clean Ice glaciers visualization")
                    ], className='navbar-header'),
                ], className='container-fluid'),
            ], className='navbar navbar-inverse'),
        ], className='p-3 bg-dark text-white'),
    ], className='container'),

    html.Div([
        html.Div([
            html.Div([
                html.H3('Model Parameters'),
                dcc.Markdown([

                ], id='model-parameters'),
            ], className='col-md-3'),

            html.Div([
                html.Div([
                    html.H3('Test set prediction'),
                ], className='text-justify'),
                html.Div([
                    dash_table.DataTable(
                        id = 'main-table',
                        data = df.to_dict('records'), 
                        columns = [{"name": i, "id": i} for i in df.columns if i != 'id'],
                        fixed_rows={'headers': True},
                        sort_action='native',
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        style_header={
                            'backgroundColor': 'rgb(30, 30, 30)',
                            'color': 'white'
                        },
                        style_data={
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white'
                        },
                    )
                ])
            ], className='col-md-9'),
        ], className='row py-lg-5'),

        html.Div([
            html.Div([
                html.Div([
                    html.H5([
                        "Input Image"
                    ], className='')
                ], className=''),
                html.Div(
                    id='x-div',
                    className=''
                )
            ], className='col-sm'),

            html.Div([
                html.Div([
                    html.H5([
                        "True Labels"
                    ], className='')
                ], className=''),
                html.Div(
                    id='y-true-div',
                    className=''
                )
            ], className='col-sm'),

            html.Div([
                html.Div([
                    html.H5([
                        "Prediction"
                    ], className='')
                ], className=''),
                html.Div(
                    id='y-pred-div',
                    className='',
                )
            ], className='col-sm'),
        ], className='row')
    ], className='container')
    ], className='main')
    return layout