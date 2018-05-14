"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You may be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mjanggo` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``janggo.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``janggo.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import argparse
import base64
import glob
import os
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

PARSER = argparse.ArgumentParser(description='Command description.')
PARSER.add_argument('-path', dest='janggo_results',
                    default=os.path.join(os.path.expanduser("~"),
                                         'janggo_results'),
                    help="Janggo results path.")

args = PARSER.parse_args()

app = dash.Dash('Janggo')
app.title = 'Janggo'
app.config['suppress_callback_exceptions'] = True


def serve_layer():
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.Nav(html.Div(dcc.Link(html.H2('Janggo'), href='/',
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
    if pathname == '/':
        print(args.janggo_results)
        files = glob.glob(os.path.join(args.janggo_results, 'models', '*.png'))
        if not files:
            return html.Div([
                html.P('The directory "{}" appears to be empty.'
                       .format(args.janggo_results))])
        encoding = {os.path.basename(os.path.splitext(name)[0]):
                    base64.b64encode(open(name, 'rb').read())
                    for name in files}

        return html.Div([
            html.Table(
                [html.Tr([html.Th('Model name'), html.Th('Architecture')])] +
                [html.Tr([dcc.Link(html.Td(name), href='/{}'.format(name)),
                 html.Td(html.Img(id=name, width='50%',
                                  src='data:image/png;base64,{}'.format(
                                      encoding[name].decode())))]) for name in encoding],
                className='table table-bordered')])
    else:
        pathlen = len(os.path.join(args.janggo_results,
                                   'evaluation', pathname[1:])) + 1
        files = glob.glob(os.path.join(args.janggo_results,
                                       'evaluation', pathname[1:], '*.png'))
        files += glob.glob(os.path.join(args.janggo_results,
                                        'evaluation', pathname[1:],
                                        '*', '*.png'))
        files += glob.glob(os.path.join(args.janggo_results,
                                       'evaluation', pathname[1:], '*.tsv'))
        files += glob.glob(os.path.join(args.janggo_results,
                                        'evaluation', pathname[1:],
                                        '*', '*.tsv'))
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
    elif value.endswith('tsv'):

        return html.Div([
        html.Div([dcc.Dropdown(id='operation',
                     options=[{'label': x,
                               'value': x} for x in ['tsne', 'svd', 'pca']],
                     value='svd'),
        dcc.Graph(id='scatter'), dcc.Graph(id='features')],
                 style={'width': '100%',
                        'display': 'inline-block',
                        'padding': '0 20'}),

        ])
    else:
        return html.P('Cannot find action for {}'.format(value))



@app.callback(
    dash.dependencies.Output('scatter', 'figure'),
    [dash.dependencies.Input('tag-selection', 'value'),
     dash.dependencies.Input('operation', 'value')])
def update_scatter(filename, operation):
    df = pd.read_csv(filename, sep='\t', header=[0,1,2])

    # if 'annot' in df use coloring
    if 'annot' in df:
        color = df.pop('annot').values
    else:
        color = 'blue'
    marker = dict(size=1, opacity=.5,
                  color=color,
                  colorscale='Viridis')
    if operation == 'svd':
        u, d, v = svd(df, full_matrices=False)
        trdf = np.dot(u[:,:3], np.diag(d[:3]))
        print('u', u.shape)
        print('d', d.shape)
        print('v', v.shape)
        print('trdf', trdf.shape)
    elif operation == 'pca':
        pca = PCA(n_components=3)
        trdf = pca.fit_transform(df)
    else:
        tsne = TSNE(n_components=3)
        trdf = tsne.fit_transform(df)

    return {'data': [
        dict(x=trdf[:,0],
             y=trdf[:,1],
             z=trdf[:,2],
             mode='markers',
             type='scatter3d',
             marker=marker)],
        'layout': go.Layout(
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            hovermode='closest',
            scene = dict(
                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0.08, y=2.2, z=0.08)
                )
            ),
            plot_bgcolor = '#E5E5E5',
            paper_bgcolor = '#E5E5E5'
        )
    }



@app.callback(
    dash.dependencies.Output('features', 'figure'),
    [dash.dependencies.Input('tag-selection', 'value'),
     dash.dependencies.Input('operation', 'value')])
def update_features(value, feature):
    df = pd.read_csv(value, sep='\t', header=[0,1,2])
    print('update_features')
    linedict = {}
    dimlist = []
    if 'annot' in df:
        color = df.pop('annot').values
    else:
        color = 0
    linedict=dict(color=color, colorscale='Viridis')

    for f in df:
        dimlist.append(dict(range = [df[f].min(), df[f].max()],
                            label=None, values=df[f]))
    return {'data': [go.Parcoords(
            line = linedict,
            dimensions = dimlist
             )],
        'layout': go.Layout(
            plot_bgcolor = '#E5E5E5',
            paper_bgcolor = '#E5E5E5'
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
    print('Welcome to janggo (GPL-v2). Copyright (C) 2017 '
          + 'Wolfgang Kopp.')
    app.run_server()
