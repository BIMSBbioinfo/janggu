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
import dash
import dash_core_components as dcc
import dash_html_components as html
import argparse

PARSER = argparse.ArgumentParser(description='Command description.')
PARSER.add_argument('-path', dest='janggo_results', 
                    default=os.path.join(os.path.expanduser("~"), 
                                         'janggo_results'),
                    help="Janggo results path.")

args = PARSER.parse_args(args=args)

app = dash.Dash()

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        files = glob.glob(args.janggo_results)
        return html.Div([
            html.H1('Janggo', href='/')
            html.Br(),
            html.Table(
            [html.Tr([html.Th('Model')]),
             [html.Tr(dcc.Link(name, href='/{}'.format(name)) for name in files] 
            ])])
    else:
        return html.Div([html.H3('You are on page {}'.format(pathname))])


def main(args=None):
    print('Welcome to janggo (GPL-v2). Copyright (C) 2017 '
          + 'Wolfgang Kopp.')
    app.run_server()
