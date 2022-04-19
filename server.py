import dash
import dash_bootstrap_components as dbc



app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE],
    prevent_initial_callbacks=False,
)

app.config.suppress_callback_exceptions = True
app.title = 'dataEng'

server = app.server



