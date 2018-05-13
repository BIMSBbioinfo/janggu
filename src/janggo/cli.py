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

import os
import glob
import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import argparse

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
        #dcc.Link(html.H1('Janggo'), href='/'),
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
        if len(files) == 0:
            return html.Div([
            html.P('The directory "{}" appears to be empty.'.format(args.janggo_results))])
        encoding = {os.path.basename(os.path.splitext(name)[0]):
                    base64.b64encode(open(name, 'rb').read()) for name in files}
        print(files)
        return html.Div([
            html.Table(
            [html.Tr([html.Th('Model name'), html.Th('Architecture')])]
             +
            [html.Tr([dcc.Link(html.Td(name), href='/{}'.format(name)),
             html.Td(html.Img(id=name, width='50%',
              src='data:image/png;base64,{}'.format(encoding[name].decode())))]) for name in encoding],
            className='table table-bordered')])
    else:
        pathlen = len(os.path.join(args.janggo_results, 'evaluation', pathname[1:])) + 1
        files = glob.glob(os.path.join(args.janggo_results, 'evaluation', pathname[1:], '*.png'))
        files += glob.glob(os.path.join(args.janggo_results, 'evaluation', pathname[1:], '*', '*.png'))


        return html.Div([html.H3('Model: {}'.format(pathname[1:])),
        dcc.Dropdown(id='tag-selection',
    options=[{'label': f[pathlen:], 'value': f} for f in files],
    value=files[0]
),    html.Div(id='output-plot')
])

@app.callback(
    dash.dependencies.Output('output-plot', 'children'),
    [dash.dependencies.Input('tag-selection', 'value')])
def update_output(value):
    img = base64.b64encode(open(value, 'rb').read())
    return html.Img( width='100%',
     src='data:image/png;base64,{}'.format(img.decode()))


external_css = ["https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css",
                "https://codepen.io/chriddyp/pen/bWLwgP.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["http://code.jquery.com/jquery-3.3.1.min.js",
               "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})

def main(args=None):
    print('Welcome to janggo (GPL-v2). Copyright (C) 2017 '
          + 'Wolfgang Kopp.')
    app.run_server()
