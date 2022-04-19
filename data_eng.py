# index page
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc


from server import app, server

# app pages
from pages import (create, about)


header = dbc.Container(
    [
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("About", href="/about"))
            ],
        )
    ]
)

app.layout = html.Div(
    [
        dbc.NavbarSimple([header], className="mb-5", id='nav-bar',
                         brand="dataEng",
                         brand_href="/",
                         color="light",
                         dark=False,
                         ),
        html.Div(id='alert'),
        dbc.Container(
            id='page-content'
        ),
        # create the session store
        dcc.Store(id='session', storage_type='session'),
        dcc.Location(id='base-url', refresh=True)
    ]
)

@app.callback(
    [Output('page-content', 'children'),
     Output('alert', 'children')],
    [Input('base-url', 'pathname')])
def router(pathname):
    page_content = None
    alert = None
    if pathname == '/':
        page_content = create.layout()
    elif pathname == '/about':
        page_content = about.layout()
    return page_content, alert

if __name__ == '__main__':
    app.run_server(debug=True)


