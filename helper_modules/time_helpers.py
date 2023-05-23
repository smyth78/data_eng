import numpy as np
from scipy import signal

from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc


from helper_modules.constants import *
from helper_modules.numerical_helpers import *
from helper_modules.alerts import *

from server import app

def time_creation_div(index, chosen_func):
    chosen_func = 'poly2' if chosen_func == [] else chosen_func[0]
    # generate the function dropdown list and incremeent index
    # eventually combination of 2 funcs
    image_link = chosen_func + '.png'
    function_drop_down_div, index = make_func_drop_down_list(index, chosen_func, image_link)
    # serve static image of fucntion with parameters
    # create input boxes for each parameter
    num_of_params = get_num_params(chosen_func)
    parameters_inputs = func_params(num_of_params)
    number_form_div = make_number_type_div(index)
    dom_div = make_domain_div(index)

    div = dbc.Row([
        dbc.Col(function_drop_down_div, width=4),
        dbc.Col(parameters_inputs, width=2),
        dbc.Col(dom_div, width=3),
        dbc.Col(number_form_div, width=3),
    ], {'type': 'de-num-input-div', 'index': index})
    index += 1
    return div, index, None

    # allow for equal spaced points or random and custom domain end points
    # if trig rad/deg


def make_func_drop_down_list(index, chosen_func, image_link):
    func_options = [{"label": i, "value": i} for i in FUNCTIONS]
    drop_down_div = [
                    dbc.Row([
                        dbc.Label("Choose function..."),
                        dbc.Select(
                            id={'type': 'de-time-func', 'index': index},
                            options=func_options,
                            value=chosen_func
                        ),
                    ]),
                    dbc.Row([
                            html.Img(src=app.get_asset_url(image_link))
                        ], style={'margin-top': ROW_MARGIN}),
                    dbc.Row([dbc.Col([
                                dbc.Label("Noise..."),
                                dcc.RangeSlider(
                                    min=0,
                                    max=5,
                                    step=0.01,
                                    marks={
                                        0: 'Less noise',
                                        5: 'More noise'
                                    },
                                    value=[0],
                                ),
                            ], width=6)], style={'margin-top': ROW_MARGIN}),
                    dbc.Row('Name the x-axis...', style={'margin-top': ROW_MARGIN}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("x-name..."),
                            dbc.Input(value='x', size="sm", type='text',
                                      id={'type': 'de-time-x-name', 'index': index}, style={"width": 100})
                        ], width=6),
                    ], style={'margin-top': ROW_MARGIN})
        ]
    index += 1
    return drop_down_div, index


def func_params(number):
    params = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    col = [dbc.Row('Parameter values...')]
    for i in range(number):
        col.append(dbc.Row([
            dbc.Col(dbc.Label(params[i]), width=2),
            dbc.Col(dbc.Input(type="number", value=1), width=6),
        ]))
    return col


def get_num_params(func):
    index_wanted = FUNCTIONS.index(func)
    return PARAMS[index_wanted]


def make_domain_div(index):
    col = [
        html.Div('Define domain...'),
        set_domain_div(index),
        dom_tick_space(index),
        # rad_deg(index)
    ]
    return col


def rad_deg(index):
    rad_deg_div = dbc.Row([
        dbc.Col([
            dbc.Label("If trig..."),
            dbc.RadioItems(
                id={'type': 'de-time-rad-deg-choice', 'index': index},
                inline=True,
                options=[
                    {'label': 'Radians', 'value': 'rad'},
                    {'label': 'Degrees', 'value': 'deg'},
                ],
                value='deg',
                style={'margin': '0 0px 10px'}
            ),
        ])
    ], style={'margin-top': ROW_MARGIN})
    return rad_deg_div


def set_domain_div(index):

    set_dom_div = dbc.Row([
        dbc.Col([
            dbc.Label("Min..."),
            dbc.Input(value=0, size="sm", type='number',
                      id={'type': 'de-time-dom-min', 'index': index}, style={"width": 100})
        ], width=6),
        dbc.Col([
            dbc.Label("Max..."),
            dbc.Input(value=10, size="sm", type='number',
                      id={'type': 'de-time-dom-max', 'index': index}, style={"width": 100})
        ], width=6)
    ])
    return set_dom_div


def dom_tick_space(index):
    dom_tick_div = dbc.Row([
        dbc.Col([
            dbc.Label("X - tick spacing..."),
            dbc.RadioItems(
                id={'type': 'de-time-tick-spacing-choice', 'index': index},
                inline=True,
                options=[
                    {'label': 'Equal', 'value': 'equal'},
                    {'label': 'Random', 'value': 'random'},
                ],
                value='equal',
                style={'margin': '0 0px 10px'}
            ),
        ])
    ], style={'margin-top': ROW_MARGIN})
    return dom_tick_div


def parse_func_info(input_div):
    is_valid = True
    func = input_div[0]['props']['children'][0]['props']['children'][1]['props']['value']
    noise = input_div[0]['props']['children'][2]['props']['children'][0]['props']['children'][1]['props']['value'][0]
    x_name = input_div[0]['props']['children'][4]['props']['children'][0]['props']['children'][1]['props']['value']
    x_name = x_name if x_name is not '' and x_name is not None else 'x'
    params = []
    param_div = input_div[1]['props']['children']
    for i in range(1, len(param_div)):
        try:
            params.append(param_div[i]['props']['children'][1]['props']['children']['props']['value'])
        except KeyError:
            is_valid = False
            return None, missing_parameters, is_valid

    dom_min = input_div[2]['props']['children'][1]['props']['children'][0]['props']['children'][1]['props']['value']
    dom_max = input_div[2]['props']['children'][1]['props']['children'][1]['props']['children'][1]['props']['value']

    x_ticks = input_div[2]['props']['children'][2]['props']['children'][0]['props']['children'][1]['props']['value']
    # angle_type = input_div[2]['props']['children'][3]['props']['children'][0]['props']['children'][1]['props']['value']
    all_inputs = [func, noise, dom_min, dom_max, x_ticks]
    if dom_min >= dom_max:
        is_valid = False
        return None, dom_max_error, is_valid
    alert = missing_parameters if any(x is None for x in all_inputs) else no_update
    return [[func, noise, x_name], params, [dom_min, dom_max, x_ticks]], alert, is_valid


def make_times_data_list(func_info, n):
    alert = no_update
    is_valid = True
    if func_info[2][2] == 'equal':
        x_values = np.linspace(func_info[2][0], func_info[2][1], num=n, endpoint=False)
    else:
        x_values = []
        for i in range(n):
            x_values.append(np.random.uniform(func_info[2][0], func_info[2][1]))

    created_data, smooth_data = create_data(func_info[0][0], x_values, func_info[1], n, func_info[0][1])
    return [created_data, smooth_data], alert, is_valid


def create_data(func, x_vals, params, n, noise):
    # is_deg = True if deg == 'deg' else False
    y_vals = []
    actual_y = []
    actual_x = np.linspace(min(x_vals), max(x_vals), num=1000) if len(x_vals) < 1000 else x_vals
    # need to create 2 sets of values...one with the chosen x_vals and one with at least 1000 x_vals so get somooth curve
    for x_val in x_vals:
        if func == 'poly0':
            y_vals.append(poly_zero(params[0]))
        elif func == 'poly1':
            y_vals.append(poly_one(x_val, params[0], params[1]))
        elif func == 'poly2':
            y_vals.append(poly_two(x_val, params[0], params[1], params[2]))
        elif func == 'poly3':
            y_vals.append(poly_three(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'sin':
            y_vals.append(sin_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'cos':
            y_vals.append(cosine_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'tan':
            y_vals.append(tan_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'ln':
            y_vals.append(ln_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'exp':
            y_vals.append(exp_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'logistic':
            y_vals.append(logistic_x(x_val, params[0], params[1], params[2], params[3]))

    # this set is for the 'smooth' curve
    for x_val in actual_x:
        if func == 'poly0':
            actual_y.append(poly_zero(params[0]))
        elif func == 'poly1':
            actual_y.append(poly_one(x_val, params[0], params[1]))
        elif func == 'poly2':
            actual_y.append(poly_two(x_val, params[0], params[1], params[2]))
        elif func == 'poly3':
            actual_y.append(poly_three(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'sin':
            actual_y.append(sin_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'cos':
            actual_y.append(cosine_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'tan':
            actual_y.append(tan_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'ln':
            actual_y.append(ln_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'exp':
            actual_y.append(exp_x(x_val, params[0], params[1], params[2], params[3]))
        elif func == 'logistic':
            actual_y.append(logistic_x(x_val, params[0], params[1], params[2], params[3]))

    y_vals_noise = []
    amp = max(y_vals) - min(y_vals) / 2
    for y_val in y_vals:
        y_vals_noise.append(apply_noise_to_point(y_val, noise, amp))
    return [x_vals, y_vals_noise], [actual_x, actual_y]


def poly_zero(a):
    return a


def poly_one(x, a, b):
    return a * x + b


def poly_two(x, a, b, c):
    return a * x ** 2 + b * x + c


def poly_three(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def sin_x(x, a, b, c, d):
    f_x = np.deg2rad(b * (x + c))
    return a * np.sin(f_x) + d


def cosine_x(x, a, b, c, d):
    return a * np.cos(np.deg2rad(b * (x + c))) + d


def tan_x(x, a, b, c, d):
    return a * np.tan(np.deg2rad(b * (x + c))) + d


def exp_x(x, a, b, c, d):
    return a * np.exp(b * (x + c)) + d


def ln_x(x, a, b, c, d):
    return a * np.log(b * (x + c)) + d

def logistic_x(x, a, b, c, d):
    return a * (1 /(1 + np.exp(-1*b * (x + c)))) + d


def apply_noise_to_point(y_val, noise, amp):
    actual_noise = np.random.normal(0, 1)
    factor = noise * amp / 5
    actual_noise *= factor
    add = np.random.randint(0, 2)
    if add == 0:
        y_val += actual_noise
    else:
        y_val -= actual_noise
    return y_val


def create_times_div(new_pd, x_name, y_name, x_smooth, y_smooth):
    x_len = len(new_pd)
    # draw data
    figure = px.scatter(new_pd, x=x_name, y=y_name)
    # draw actual function ith signal smoother
    figure.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='f(x)'))
    figure['layout']['margin'] = {'l': 5, 'r': 5, 'b': 5, 't': 5}
    figure.update_layout(showlegend=False, template='simple_white')
    graph = dcc.Graph(figure=figure)
    table = get_current_df_as_table([{"name": i, "id": i} for i in new_pd.columns], new_pd.to_dict('records'))
    cols = [
        dbc.Col(graph, width=8),
        dbc.Col(table, width=4)
    ]
    return cols
