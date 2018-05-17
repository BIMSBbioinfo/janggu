"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You may be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mjanggu` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``janggu.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``janggu.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import argparse
import base64
import glob
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.linalg import svd
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PARSER = argparse.ArgumentParser(description='Command description.')
PARSER.add_argument('-path', dest='janggu_results',
                    default=os.path.join(os.path.expanduser("~"),
                                         'janggu_results'),
                    help="Janggu results path.")

args = PARSER.parse_args()

app = dash.Dash('Janggu')
app.title = 'Janggu'
app.config['suppress_callback_exceptions'] = True


def serve_layer():
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.Nav(html.Div(dcc.Link(html.H2('Janggu'), href='/',
                                   className='navbar-brand'),
                          className='navbar-header'),
                 className='navbar navbar-default navbar-expand-lg navbar-dark'),
        html.Br(),
        html.Div(id='page-content')
    ], className='container')


app.layout = serve_layer()


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname is None:
        return html.Div([])
    elif pathname == '/':
        print(args.janggu_results)
        files = glob.glob(os.path.join(args.janggu_results, 'models', '*.png'))
        if not files:
            return html.Div([
                html.P('The directory "{}" appears to be empty.'
                       .format(args.janggu_results))])
        encoding = {os.path.basename(os.path.splitext(name)[0]):
                    base64.b64encode(open(name, 'rb').read())
                    for name in files}

        return html.Div([
            html.Table(
                [html.Tr([html.Th('Model name'), html.Th('Architecture')])] +
                [html.Tr([dcc.Link(html.Td(name), href='/{}'.format(name)),
                          html.Td(html.Img(
                              id=name, width='50%',
                              src='data:image/png;base64,{}'.format(
                                  encoding[name].decode())))])
                 for name in encoding],
                className='table table-bordered')])
    else:
        pathlen = len(os.path.join(args.janggu_results,
                                   'evaluation', pathname[1:])) + 1
        files = glob.glob(os.path.join(args.janggu_results,
                                       'evaluation', pathname[1:], '*.png'))
        files += glob.glob(os.path.join(args.janggu_results,
                                        'evaluation', pathname[1:],
                                        '*', '*.png'))
        files += glob.glob(os.path.join(args.janggu_results,
                                        'evaluation', pathname[1:], '*.ply'))
        files += glob.glob(os.path.join(args.janggu_results,
                                        'evaluation', pathname[1:],
                                        '*', '*.ply'))
        if not files:
            return html.Div([html.H3('No figures available for {}'.format(pathname[1:]))])

        return html.Div([html.H3('Model: {}'.format(pathname[1:])),
                         dcc.Dropdown(id='tag-selection',
                                      options=[{'label': f[pathlen:],
                                                'value': f} for f in files],
                                      value=files[0]),
                         html.Div(id='output-plot')])


@app.callback(
    dash.dependencies.Output('output-plot', 'children'),
    [dash.dependencies.Input('tag-selection', 'value')])
def update_output(value):

    if value.endswith('png'):
        # display the png images directly
        img = base64.b64encode(open(value, 'rb').read())
        return html.Img(width='100%',
                        src='data:image/png;base64,{}'.format(img.decode()))
    elif value.endswith('ply'):
        dfheader = pd.read_csv(value, sep='\t', header=[0], nrows=0)
        # read the annotation for the dropdown
        annot = []
        for col in dfheader.columns:
            if col[:len('annot.')] == 'annot.':
                annot.append(col[len('annot.'):])

        return html.Div([
            html.Div([
                dcc.Dropdown(id='xaxis',
                             options=[{'label': x,
                                       'value': x}
                                      for x in ['Component {}'.format(i)
                                                for i in [1, 2, 3]]],
                             value='Component 1'),
                dcc.Dropdown(id='yaxis',
                             options=[{'label': x,
                                       'value': x}
                                      for x in ['Component {}'.format(i)
                                                for i in [1, 2, 3]]],
                             value='Component 2'),
                dcc.Graph(id='scatter')],
                     style={'width': '100%',
                            'display': 'inline-block',
                            'padding': '0 20'}),
            html.Div([
                dcc.Dropdown(id='operation',
                             options=[{'label': x,
                                       'value': x}
                                      for x in ['tsne', 'svd', 'pca']],
                             value='svd'),
                dcc.Dropdown(id='annotation',
                             options=[{'label': x,
                                       'value': x} for x in ['None'] + annot],
                             value='None'),
                dcc.Graph(id='features')],
                     style={'width': '100%',
                            'display': 'inline-block',
                            'padding': '0 20'}),
        ], style={'columnCount': 2})

    # else the value is not known
    return html.P('Cannot find action for {}'.format(value))


@app.callback(
    dash.dependencies.Output('scatter', 'figure'),
    [dash.dependencies.Input('tag-selection', 'value'),
     dash.dependencies.Input('operation', 'value'),
     dash.dependencies.Input('xaxis', 'value'),
     dash.dependencies.Input('yaxis', 'value'),
     dash.dependencies.Input('annotation', 'value')])
def update_scatter(filename, operation, xaxis_label, yaxis_label, annotation):
    df = pd.read_csv(filename, sep='\t', header=[0])

    # if 'annot' in df use coloring
    colors = pd.Series(['blue'] * df.shape[0])
    for col in df:
        if col[:len('annot.')] == 'annot.':
            if annotation == col[len('annot.'):]:
                colors = df[col]
            df.pop(col)
        if col == 'row_names':
            text = df.pop(col)

    marker = dict(size=3, opacity=.5)
    if operation == 'svd':
        u, d, v = svd(df, full_matrices=False)
        trdf = np.dot(u[:, :3], np.diag(d[:3]))
    elif operation == 'pca':
        pca = PCA(n_components=3)
        trdf = pca.fit_transform(df)
    else:
        tsne = TSNE(n_components=3)
        trdf = tsne.fit_transform(df)
    trdf = pd.DataFrame(trdf, columns=['Component {}'.format(i)
                                       for i in [1, 2, 3]])

    print(colors.unique())
    data = []
    for c in colors.unique():
        data.append(
            go.Scatter(
                x=trdf[xaxis_label][colors == c],
                y=trdf[yaxis_label][colors == c],
                text=text,
                name=c,
                customdata=colors[colors == c].index,
                mode='markers',
                marker=dict(size=3, opacity=.5)))

    return {'data': data,
            'layout': go.Layout(
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                xaxis={'title': xaxis_label},
                yaxis={'title': yaxis_label},
                hovermode='closest',
                plot_bgcolor='#E5E5E5',
                paper_bgcolor='#E5E5E5'
            )
           }


@app.callback(
    dash.dependencies.Output('features', 'figure'),
    [dash.dependencies.Input('tag-selection', 'value'),
     dash.dependencies.Input('operation', 'value'),
     dash.dependencies.Input('scatter', 'selectedData')])
def update_features(value, feature, selectedData):
    df = pd.read_csv(value, sep='\t', header=[0])
    df.reindex_axis(sorted(df.columns), axis=1)

    print('update_features')

    for col in df:
        if col[:len('annot')] == 'annot':
            df.pop(col)
        if col == 'row_names':
            df.pop(col)

    print(selectedData)
    df = df.apply(zscore)
    if selectedData is not None:
        selected_idx = [datum['customdata'] for datum in selectedData['points']]
        df = df.iloc[selected_idx, :]
        print(selected_idx)
        print(df.shape)

    mean = df.mean().values
    std = df.std().values

    return {'data': [go.Scatter(
        x=list(range(len(mean))),
        y=mean,
        mode='line',
        text=df.columns,
        error_y=dict(
            type='percent',
            value=std,
            thickness=1,
            width=0,
            color='#444',
            opacity=0.8
        ))],
            'layout': go.Layout(
                xaxis={'title': 'Features'},
                plot_bgcolor='#E5E5E5',
                paper_bgcolor='#E5E5E5',
                yaxis={'showgrid': False, 'title': 'Activities (z-score)'}
            )
           }


external_css = ["https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css",
                "https://codepen.io/chriddyp/pen/bWLwgP.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["http://code.jquery.com/jquery-3.3.1.min.js",
               "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})


def main():
    """cli entry"""
    print('Welcome to janggu (GPL-v2). Copyright (C) 2017 '
          + 'Wolfgang Kopp.')
    app.run_server()
