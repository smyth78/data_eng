import json
from scipy.stats import skewnorm
from sigfig import round
import math
import random
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from scipy.stats import spearmanr
from numpy.linalg import LinAlgError

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc
from dash import no_update
import plotly.subplots as splts
import plotly.graph_objs as go
import plotly.express as px

from helper_modules.alerts import *
from helper_modules.constants import *
from helper_modules.general_helpers import *
from helper_modules.categorical_helpers import get_dynamic_radio as gdr


def num_assoc_div(index, current_choice):
    current_choice = ['none'] if current_choice == [] else current_choice
    div = html.Div([
        dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Row([html.H5('Type of association for the new feature...')]),
                dbc.Row([
                    dbc.RadioItems(
                        id={'type': 'de-num-assoc-type', 'index': index},
                        inline=True,
                        options=[
                            {'label': 'Category', 'value': 'cat'},
                            {'label': 'Numerical', 'value': 'num'},
                            {'label': 'Numerical split by cat', 'value': 'num-cat'},
                            {'label': 'None', 'value': 'none'},
                        ],
                        value=current_choice[0],
                        style={'margin': '0 0px 10px'})]),
            ], id='de-num-assoc-type-div'),
    ])
    ]),
    ])
    return div


def correlation_inputs(cats, index, cat_assoc):
    is_assoc = False if 'no' in cats else True
    corr_div = []
    for cat in cats:
        cat_div = []
        cat_div.append(
            correlation_degree_input(index, cat, is_assoc, cat_assoc)
        )
        cat_div.append(
            correlation_size_input(index, cat, is_assoc)
        )
        index += 1
        cat_div.append(dbc.Col(THEME_BREAK, width={"size": 6, "offset": 3})) if is_assoc else None
        corr_div.append(dbc.Row(cat_div))
    return corr_div

def dist_drop_down(cats, index, dist_type, cat_assoc):
    is_assoc = False if 'no' in cats else True
    dist_div = []
    for cat in cats:
        cat_div = []
        cat_div.append(dbc.Col([
            dbc.Label(html.Strong(cat)) if is_assoc else dbc.Label("Choose distribution type..."),
            dbc.Select(

                id={'type': 'de-num-dist-type', 'index': index},
                options=[
                    {"label": "Uniform", "value": "uniform"},
                    {"label": "Normal", "value": "normal"},
                    {"label": "Skew normal", "value": "skew"},
                    {"label": "Exponential", "value": "exponential"},
                    {"label": "Percentiles", "value": "percentiles"}
                ],
                value='normal'
            ),
            ], width=4)
        )
        if 'normal' in dist_type:
            cat_div.append(dbc.Col([
                        normal_params()
                        ], id={'type': 'de-num-para-input', 'index': index}, width=8),
                    )
        elif 'uniform' in dist_type:
            cat_div.append(dbc.Col([
                        uniform_params()
                    ], id={'type': 'de-num-para-input', 'index': index}, width=8)
            )
        elif 'skew' in dist_type:
            cat_div.append(dbc.Col([
                        skew_params()
                    ], id={'type': 'de-num-para-input', 'index': index}, width=8),
            )
        elif 'exponential' in dist_type:
            cat_div.append(dbc.Col([
                        exponential_params()
                    ], id={'type': 'de-num-para-input', 'index': index}, width=6)
            )
        elif 'percentiles' in dist_type:
            cat_div.append(dbc.Col([
                        percentiles_params()
                    ],
            id={'type': 'de-num-para-input', 'index': index}, width=8),
            )
        else:
            cat_div.append(None)
        index += 1
        cat_div.append(dbc.Col(THEME_BREAK, width={"size": 6, "offset": 3})) if is_assoc else None
        dist_div.append(dbc.Row(cat_div))
    return dist_div


def deg_acc_div(index):
    deg_acc_div = html.Div([
        dbc.Label("Degree of accuracy..."),
        dbc.RadioItems(
            id={'type': 'de-num-deg-acc-choice', 'index': index},
            inline=True,
            options=[
                {'label': 'Decimal places', 'value': 'dp'},
                {'label': 'Significant figures', 'value': 'sf'},
            ],
            value='sf',
            style={'margin': '0 0px 10px'}
        ),
        dbc.Input(value=3, size="sm", type='number', min=1, max=10, step=1, id={'type': 'de-num-deg-acc-size', 'index': index}, style={"width": 75})
    ])
    return deg_acc_div

def number_type_div(index):
    number_type_div = html.Div([
        dbc.Label("Type of number..."),
        dbc.RadioItems(
            id={'type': 'de-num-type-choice', 'index': index},
            inline=True,
            options=[
                {'label': 'Floating point', 'value': 'float'},
                {'label': 'Integer', 'value': 'int'},
            ],
            value='float',
            style={'margin': '0 0px 10px'}
        ),
        dbc.Checklist(
            id={'type': 'de-num-type-non-negative', 'index': index},
            options=[
                {"label": "Non-negative?", "value": 'non-neg'},
            ],
            value=[],
            inline=True,
        ),
    ])
    return number_type_div


def correlation_size_input(index, cat, is_assoc):
    cat_string = '(' + cat + ')'
    corr_input_div = dbc.Col(
        [
            dbc.Row([
                dbc.Col([
                    dbc.Label("New mean..." + cat_string) if is_assoc else dbc.Label("Mean..."),
                    dbc.Input(type="number", name=cat),
                ], width=6),
                dbc.Col([
                    dbc.Label("New SD..." + cat_string) if is_assoc else dbc.Label("SD..."),
                    dbc.Input(type="number", name=cat),
                ], width=6),
            ]),
            THEME_BREAK if 'none' not in cat else None,
        ],
        id={'type': 'de-num-corr-size-input', 'index': index}, width=6
    )
    return corr_input_div


def correlation_degree_input(index, cat, is_assoc, cat_assoc):
    cat_string = '(' + cat + ')'
    corr_input_div = dbc.Col(
        [
            dbc.Row([
                dbc.Col([
                    dbc.Label(html.Strong(cat)) if is_assoc else dbc.Label("Correlation..."),
                    dcc.RangeSlider(
                        min=-0.99,
                        max=0.99,
                        step=0.01,
                        marks={
                            -0.99: '-ve',
                            0.99: '+ve'
                        },
                        value=[0],
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Noise..." + cat_string) if is_assoc else dbc.Label("Noise..."),
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
                ], width=6),
            ]),
        ],
        id={'type': 'de-num-corr-degree-input', 'index': index}, width=6
    )
    return corr_input_div

def num_dist_div(index, df_num, df_cat, dist_type, num_assoc, cat_assoc):
    alert = no_update
    try:
        cats = np.unique(df_cat[cat_assoc].values) if cat_assoc is not 'no' else ['no']
    except TypeError:
        print('this is the problem when the spreadhseet records an empty entry')
        alert = unknown_error
        return None, index, alert

    is_dist_choice = True if num_assoc in ['cat', 'no'] else False
    # main_row = dist_drop_down(cats, index, dist_type) if is_dist_choice else correlation_inputs(cats, index)
    main_row = dist_drop_down(cats, index, dist_type, cat_assoc) if is_dist_choice else correlation_inputs(cats, index, cat_assoc)
    title_string = get_num_title(num_assoc, cat_assoc, not is_dist_choice)
    div = html.Div([
        dbc.Row([title_string]),
        dbc.Row([
            dbc.Col(main_row, width=8),
            dbc.Col(make_number_type_div(index), width=4)
        ], id={'type': 'de-num-input-div', 'index': index}, style={'padding-top': ROW_MARGIN}),
        dbc.Row(type_of_chart_combo(index, num_assoc))
    ])
    index += 1
    return div, index, alert


def type_of_chart_combo(index, num_assoc):
    is_num_assoc = False if num_assoc == 'no' else True
    div = dbc.Col([
    dbc.Label("Output chart combination..."),
    dbc.RadioItems(
        id={'type': 'de-num-chart-combo', 'index': index},
        inline=True,
        options=[
            {'label': 'Histo/Box', 'value': 'histbox'},
            {'label': 'Boxplot(s)', 'value': 'b2bbox'},
        ],
        value='histbox',
        style={'margin': '0 0px 10px'}
    )], style={'padding-top': ROW_MARGIN, 'display': 'none' if is_num_assoc else 'block'}, width=6)
    return div


def num_creation_div(index, df_num, df_cat, num_dist_type, cat_assoc, num_assoc):
    dist_div, index, alert = num_dist_div(index, df_num, df_cat, num_dist_type, num_assoc, cat_assoc)

    cat_radio_div = html.Div([
        html.Div([gdr(df_cat, index, cat_assoc)]),
        THEME_BREAK
    ])

    num_radio_div = html.Div([
        html.Div([get_dynamic_num_radio(df_num, index, num_assoc)]),
        THEME_BREAK
    ])

    num_div = html.Div([
        cat_radio_div,
        num_radio_div,
        dist_div
    ])
    return num_div, index, alert


def get_dynamic_num_radio(df_num, index, radio_choice):
    options = [{"label": i, "value": i} for i in df_num.columns] if df_num is not None else []
    options.append({'label': 'No', 'value': 'no'})
    div = html.Div([
        dbc.Row([html.H5('Associate new feature with existing numerical feature...')]),
        dbc.RadioItems(
            id={'type': 'de-num-assoc', 'index': index},
            inline=True,
            options=options,
            value=radio_choice,
        )
    ])
    return div


def is_dynamic_input_num_div(input_str):
    is_correct_key = False
    input_list = input_str.split('.')
    try:
        dict_cb_input = json.loads(input_list[0])
        if dict_cb_input['type'] == 'de-num-input-div':
            is_correct_key = True
    except ValueError:
        print('not a json')
    return is_correct_key


def parse_number_type(input_div, param):
    is_dp = True if input_div[0][param]['props']['children']['props']['children'][0]['props'][
                             'children'][0]['props']['children'][1]['props']['value'] == 'dp' else False
    deg_of_acc = input_div[0][param]['props']['children']['props']['children'][0]['props']['children'][0]['props']['children'][2]['props']['value']
    is_float = True if input_div[0][param]['props']['children']['props']['children'][1]['props'][
                           'children'][0]['props']['children'][1]['props']['value'] == 'float' else False
    is_non_negative = False if input_div[0][param]['props']['children']['props']['children'][1]['props'][
                           'children'][0]['props']['children'][2]['props']['value'] == [] else True
    return is_dp, deg_of_acc, is_float, is_non_negative


def get_corr_and_params(input_div, num_assoc):
    is_valid = True
    alert = no_update
    corr_params = input_div[0][0]['props']['children']

    corr_cat_params = []
    for corr_param in corr_params:
        corr_cat_param = []
        try:
            cat_type = corr_param['props']['children'][0]['props']['children'][0]['props']['children'][0]['props']['children'][0]['props']['children']['props']['children']
        except TypeError:
            cat_type = 'no'
        corr_strength = corr_param['props']['children'][0]['props']['children'][0]['props']['children'][0]['props']['children'][1]['props']['value'][0]
        noise_strength = corr_param['props']['children'][0]['props']['children'][0]['props']['children'][1]['props']['children'][1]['props']['value'][0]
        corr_cat_param.append([cat_type, corr_strength, noise_strength])

        new_mean = corr_param['props']['children'][1]['props']['children'][0]['props']['children'][0]['props']['children'][1]['props']['value']
        new_sd = corr_param['props']['children'][1]['props']['children'][0]['props']['children'][1]['props']['children'][1]['props']['value']
        corr_cat_param.append([new_mean, new_sd])
        corr_cat_params.append(corr_cat_param)
        if None in [cat_type, corr_strength, noise_strength, new_mean, new_sd]:
            is_valid = False
            alert = missing_parameters
    return corr_cat_params, is_valid, alert


def get_dist_and_params(input_div):
    is_valid = True
    alert = no_update
    dist_divs = input_div[0]['props']['children']
    dist_info = []
    for dist_div in dist_divs:
        try:
            cat_type = dist_div['props']['children'][0]['props']['children'][0]['props']['children']['props']['children']
        except TypeError:
            cat_type = 'no'
        dist_type = dist_div['props']['children'][0]['props']['children'][1]['props']['value']
        params_list_divs = dist_div['props']['children'][1]['props']['children']['props']['children']
        params_list = []
        for params in params_list_divs:
            try:
                params_list.append(params['props']['children'][1]['props']['value'])
            except KeyError:
                alert = missing_parameters
                is_valid = False
        dist_info.append([[cat_type, dist_type], params_list])
    return dist_info, is_valid, alert


def verify_num_dist_input_data(dist_info, is_num_assoc):
    is_valid = True
    alert = None
    if is_num_assoc:
        for dist in dist_info:
            if dist[1][1] <= 0:
                alert = sd_pos
                is_valid = False
    else:
        for dist in dist_info[0]:
            if dist[0][1] in ['uniform', 'skew']:
                if dist[1][0] >= dist[1][1]:
                    alert = max_min_val
                    is_valid = False
            elif 'normal' == dist[0][1]:
                if dist[1][1] <= 0:
                    alert = sd_pos
                    is_valid = False
            elif 'exponential' == dist[0][1]:
                if dist[1][0] <= 0:
                    alert = ex_pos
                    is_valid = False
    return is_valid, alert


def get_skew(min_val, max_val, skewness, n):
    data_list = skewnorm.rvs(a=float(skewness), loc=float(max_val), size=int(n))
    data_list = (max_val - min_val) / (max(data_list) - min(data_list)) * (data_list - max(data_list)) + max_val
    return data_list


def get_uniform_dist(min, max, n):
    return np.random.uniform(min, max, n)


def get_normal_data(mean, sd, n):
    return np.random.normal(mean, sd, n)


def get_skew_normal(min, max, skewness, n):
    return get_skew(min, max, skewness, n)


def get_exponential(scale, n):
    return np.random.exponential(scale, n)


def make_num_data_list(dist_info, is_num_f_assoc, num_assoc, cat_assoc, full_df):
    alert= no_update
    is_valid = True
    data_lists = []
    if is_num_f_assoc:
        for dist in dist_info:
            dist_list = []
            filtered_df = full_df.loc[full_df[cat_assoc] == dist[0][0], num_assoc] if cat_assoc != 'no' else full_df[num_assoc]
            data_values, rho_sample, sp_rank, alert, is_valid = create_corr_data_set(filtered_df.values, dist[0][1], dist[1][0], dist[1][1], dist[0][2])
            if not is_valid:
                return None, alert, is_valid
            dist_list.append([num_assoc, cat_assoc, dist[0][0], data_values, float(rho_sample[0][1]), float(sp_rank.correlation)])
            data_lists.append(dist_list)
    else:
        for dist in dist_info[0]:
            n = dist[0][2]
            cat_name = dist[0][0]
            dist_type = dist[0][1]
            params = dist[1]
            data_values = []
            if dist_type == 'uniform':
                data_values = get_uniform_dist(params[0], params[1], n)
            elif dist_type == 'normal':
                data_values = get_normal_data(params[0], params[1], n)
            elif dist_type == 'skew':
                data_values = get_skew(params[0], params[1], params[2], n)
            elif dist_type == 'exponential':
                data_values = get_exponential(params[0], n)
            elif dist_type == 'percentiles':
                data_values = get_percentile_dataset(params, n)
            cat_list = [cat_name, data_values]
            data_lists.append(cat_list)
    return data_lists, alert, is_valid


def round_list(is_dp, acc, data_list, is_float, is_non_neg, is_num_assoc, y_vals, is_times):
    rounded_data = []
    if not is_times:
        iterable_list = data_list[0][3] if is_num_assoc else data_list[1]
    else:
        iterable_list = y_vals
    for point in iterable_list:
        point = 0 if point < 0 and is_non_neg else point
        if is_dp:
            if is_float:
                rounded_data.append(round(float(point), decimals=acc))
            else:
                rounded_data.append(round(int(point), decimals=acc))
        else:
            if is_float:
                rounded_data.append(round(float(point), sigfigs=acc))
            else:
                round_to_int = round(float(point), dec=0)
                rounded_point = int(round(round_to_int, sigfigs=acc))
                rounded_data.append(rounded_point)
    return rounded_data


def make_num_corr_box(list_of_dfs, new_f_name, num_assoc, cat_assoc, output_chart, full_df):
    pop_trace = []
    pop_split_traces = []
    number_of_dfs = len(list_of_dfs)

    distinct_cats = np.unique(full_df[cat_assoc].values) if cat_assoc != 'no' else None

    column_counter = 1
    figure_split = None
    # pop figure
    pop_df = list_of_dfs[0]
    pop_scatter = px.scatter(pop_df, x=num_assoc, y=new_f_name)
    pop_scatter['layout']['margin'] = {'l': 5, 'r': 5, 'b': 5, 't': 5}
    pop_scatter.update_layout(showlegend=False, template='simple_white')

    split_scatter_gos = []
    if number_of_dfs > 1:
        for cat in distinct_cats:
            # filtered_df_assoc = pop_df.loc[pop_df[cat_assoc] == cat, num_assoc]
            # filtered_df_new = pop_df.loc[pop_df[cat_assoc] == cat, new_f_name]
            filt_df = pop_df.loc[pop_df[cat_assoc] == cat]
            scatter = px.scatter(filt_df, x=num_assoc, y=new_f_name)
            y_title = new_f_name + ' - ' + cat + '->' + cat_assoc
            scatter.update_layout(yaxis_title=y_title)
            scatter['layout']['margin'] = {'l': 5, 'r': 5, 'b': 5, 't': 5}
            scatter.update_layout(showlegend=False, template='simple_white')

            column_counter += 1
            split_scatter_gos.append(dcc.Graph(figure=scatter))
    # title_string = 'Population summary of... ' + new_f_name if 'histbox' in output_chart else 'Summary of... ' + new_f_name
    # is_hist_cat_div = True if cat_assoc != [] and 'histbox' in output_chart else False
    div = dbc.Col([dbc.Row(html.H5('Population')),
                   dbc.Row(dcc.Graph(figure=pop_scatter)),
                   THEME_BREAK,
                   dbc.Row(html.P('Split graphs...')) if cat_assoc != 'no' else None ,
                   dbc.Row(split_scatter_gos)
                   ], width=12, style={'margin-right': COLUMN_PAD}, id={'type': 'num-div-display', 'index': 0})
    return div

def make_num_dist_box(list_of_dfs, new_f_name, cat_assoc, output_chart):
    pop_trace = []
    pop_split_traces = []
    number_of_dfs = len(list_of_dfs)

    column_counter = 1

    figure_split = None
    figure_pop_box = None
    figure_pop = None

    if 'histbox' in output_chart:
        # this is the population fig histbox
        figure_pop = splts.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.009,
                                         horizontal_spacing=0.009, row_width=[0.8, 0.2])
        figure_pop.update_layout(showlegend=False, template='simple_white')
        figure_pop['layout']['margin'] = {'l': 5, 'r': 5, 'b': 5, 't': 5}
    else:
        row_heights = make_box_figure_row_heights(number_of_dfs)
        # this is the population fig box2box
        figure_pop_box = splts.make_subplots(rows=number_of_dfs, cols=1, shared_xaxes=True, vertical_spacing=0.009,
                                         horizontal_spacing=0.009, row_heights=row_heights if number_of_dfs > 1 else [1])
        figure_pop_box.update_layout(showlegend=False, template='simple_white')
        figure_pop_box['layout']['margin'] = {'l': 5, 'r': 5, 'b': 5, 't': 5}

    num_cols = number_of_dfs - 1
    if num_cols > 0 and 'histbox' in output_chart:
        # this is the split fig only needed when we have more than one df
        figure_split = splts.make_subplots(rows=2, cols=num_cols, shared_xaxes=True, vertical_spacing=0.05,
                                           horizontal_spacing=0.2, row_width=[0.8, 0.2])
        figure_split['layout']['margin'] = {'l': 5, 'r': 5, 'b': 5, 't': 5}
        figure_split.update_layout(showlegend=False, template='simple_white')

    pop_df = list_of_dfs[0][[new_f_name, cat_assoc[0]]] if 'no' not in cat_assoc else list_of_dfs[0][[new_f_name]]
    min_pop_val = math.floor(min(pop_df[new_f_name].values))
    max_pop_val = math.ceil(max(pop_df[new_f_name].values))

    if 'histbox' in output_chart:
        for index, df in enumerate(list_of_dfs):
            is_pop_div = True if index == 0 else False
            cat_name = df.columns.values.tolist()[0]
            fig = figure_pop if is_pop_div else figure_split
            make_num_histbox_output_figure_subplots(df, fig, new_f_name, is_pop_div, cat_name, column_counter, min_pop_val, max_pop_val)
            column_counter = 0 if index == 0 else column_counter
            column_counter += 1

        pop_trace.append(get_fig_into_graph(figure_pop, 0))
        pop_split_traces.append(get_fig_into_graph(figure_split, column_counter))

    else:
        for index, df in enumerate(list_of_dfs):
            is_pop_div = True if index == 0 else False
            cat_name = df.columns.values.tolist()[0]
            make_num_box2box_output_figure_subplots(df, figure_pop_box, new_f_name, is_pop_div, cat_name,
                                                    column_counter, min_pop_val, max_pop_val, number_of_dfs)
            column_counter += 1
        pop_trace.append(get_fig_into_graph(figure_pop_box, 0))
    created_data_as_table = get_created_data_as_table([{"name": i, "id": i} for i in pop_df.columns],
                                                      pop_df.to_dict('records'))
    title_string = 'Population summary of... ' + new_f_name if 'histbox' in output_chart else 'Summary of... ' + new_f_name
    is_hist_cat_div = True if cat_assoc != [] and 'histbox' in output_chart else False
    div = dbc.Col([dbc.Row(html.H5(title_string)),
                   dbc.Row(pop_trace),
                   THEME_BREAK,
                   html.Div([
                       dbc.Row(html.H5(new_f_name + ' split by ' + cat_assoc[0] if cat_assoc != ['no'] else None)) if cat_assoc != ['no'] else None,
                       dbc.Row(pop_split_traces if cat_assoc != ['no'] else None)
                   ]) if is_hist_cat_div else None,
                   dbc.Row([
                       dbc.Col(created_data_as_table)
                   ])
                   ],
                  width=12, style={'margin-right': COLUMN_PAD}, id={'type': 'num-div-display', 'index': 0})

    return div


def add_sizes_too_dist_info(dist_info, sizes, is_cat_f_assoc, is_num_f_assoc):
    if is_num_f_assoc:
        for dist in dist_info:
            for size in sizes:
                if dist[0][0] == size[0]:
                    dist[0].append(size[1])

    else:
        for dist in dist_info[0]:
            for size in sizes:
                if dist[0][0] == size[0]:
                    dist[0].append(size[1])
    return dist_info

# def parse_param_data(params_list, size_list, is_cat_f_assoc):
#     parsed_params = []
#     percent_params = []
#     for cat in params_list:
#         cat_info = []
#         cat_params = []
#         if cat[0][1] == 'percentiles':
#             percent_params.append([[cat[1][0]], cat[1]])
#         else:
#             for param in cat:
#                 cat_info.append(param[0])
#                 cat_params.append(param[1])
#             cat_params.insert(0, [cat_names[0]])
#             parsed_params.append(cat_params)
#     # now add the size of the cat
#     chosen_params = percent_params if dist_type == 'percentiles' else parsed_params
#     for cat in chosen_params:
#         for size in size_list:
#             if is_cat_f_assoc:
#                 if cat[0][0] == size[0]:
#                     cat[0].append(size[1])
#             else:
#                 cat[0].append(size)
#     return percent_params if dist_type == 'percentiles' else parsed_params


def get_hist_go(df, feature_name, min_bin):
    hist = go.Histogram(x=df[feature_name].values, nbinsx=15,
                        xbins=dict(start=min_bin),
                        marker=dict(color=RED,
                                    line=dict(color='rgba(0, 0, 0, 0.8)')))
    return hist

def get_box_go(df, feature_name, cat_name, colour):
    box = go.Box(x=df[feature_name].values, name=cat_name, fillcolor=colour, line=dict(color='rgba(0, 0, 0, 0.8)'))
    return box


def make_num_box2box_output_figure_subplots(df, figure, new_f_name, is_pop_div, cat_name, column_counter, min, max, num_dfs):
    colour = BLUE if column_counter == 1 else RED
    figure.append_trace(
        get_box_go(df, new_f_name if is_pop_div else cat_name, 'population' if is_pop_div else cat_name, colour), row=column_counter,
        col=1)
    # update box plot axes
    figure.update_xaxes(showline=True if column_counter == num_dfs else False,
                        title_text=new_f_name if column_counter == num_dfs else None,
                        row=column_counter, col=1, range=[min, max])
    figure.update_yaxes(showline=False, row=column_counter, col=1)
    # figure.update_yaxes(row=column_counter, col=1, title_text='Population')


def make_num_histbox_output_figure_subplots(df, figure, new_f_name, is_pop_div, cat_name, column_counter, min, max):

    figure.append_trace(get_box_go(df, new_f_name if is_pop_div else cat_name, 'population' if is_pop_div else cat_name,
                                   BLUE), row=1, col=column_counter)
    figure.append_trace(get_hist_go(df, new_f_name if is_pop_div else cat_name, min), row=2, col=column_counter)
    # update box plot axes
    figure.update_xaxes(showline=False, row=1, col=column_counter)
    figure.update_yaxes(showline=False, row=1, col=column_counter)
    # update histo axes
    split_cat_x_axis_title = cat_name + ' ' + new_f_name
    figure.update_xaxes(row=2, col=column_counter,
                        title_text=new_f_name if is_pop_div else split_cat_x_axis_title,
                        range=[min, max])
    figure.update_yaxes(row=2, col=column_counter, title_text='Frequency')


def get_fig_into_graph(figure, column_counter):
    return dcc.Graph(figure=figure, id={'type': 'num-chart-display', 'index': column_counter}, className="w-100")


def get_percentile_dataset(para_list, n):
    num_of_groups = len(para_list) - 1
    group_freqs = [int(n / num_of_groups)] * num_of_groups
    remaining_recs = n % num_of_groups

    # now add the rem recs randomly
    for i in range(remaining_recs):
        random.shuffle(group_freqs)
        group_freqs[0] = group_freqs[0] + 1

    i = 0
    data_list = np.array([])
    while i < num_of_groups:
        # append the min value in group
        data_list = np.append(data_list, float(para_list[i]))
        # append n-2 random values between the percentiels
        data = np.random.uniform(float(para_list[i]), float(para_list[i + 1]), group_freqs[i] - 2)
        data_list = np.append(data_list, data)
        # append the upper percentile
        data_list = np.append(data_list, float(para_list[i + 1]))
        i += 1
    data_list = data_list.flatten()
    data_list = list(data_list)
    random.shuffle(data_list)
    return data_list


def make_box_figure_row_heights(num_bp):
    row_width = [0.5]
    rem_row_width = 0.5 / num_bp
    for i in range(num_bp - 1):
        row_width.append(rem_row_width)
    return row_width


def make_number_type_div(index):
    row = dbc.Row([
        dbc.Col([
                deg_acc_div(index)
            ], width=6),
        dbc.Col([
                number_type_div(index)
            ], width=6),
    ])
    return row



def normal_params():
    return dbc.Row([
        dbc.Col([
            dbc.Label("Mean..."),
            dbc.Input(type="number"),
        ], width=6),
        dbc.Col([
            dbc.Label("SD..."),
            dbc.Input(type="number"),
        ], width=6),
    ])


def uniform_params():
    return dbc.Row([
        dbc.Col([
            dbc.Label("Min value..."),
            dbc.Input(type="number"),
        ], width=6),
        dbc.Col([
            dbc.Label("Max value..."),
            dbc.Input(type="number"),
        ], width=6),
    ])


def skew_params():
    return dbc.Row([
        dbc.Col([
            dbc.Label("Min value..."),
            dbc.Input(type="number"),
        ], width=4),
        dbc.Col([
            dbc.Label("Max value..."),
            dbc.Input(type="number"),
        ], width=4),
        dbc.Col([
            dbc.Label("Skewness..."),
            dbc.Input(type="number"),
        ], width=4),
    ])


def exponential_params():
    return dbc.Row([
        dbc.Col([
            dbc.Label("Scale..."),
            dbc.Input(type="number", min=0),
        ], width=6),
    ])


def percentiles_params():
    return dbc.Row([
        dbc.Col([
            dbc.Label("Percentiles..."),
            dbc.Input(),
        ], width=12)
    ])


def parse_percentiles(dist_info):
    is_valid = True
    alert = None
    parsed_dist_info = []
    for dist in dist_info:
        if 'percentiles' in dist[0][1]:
            num_params = dist[1][0].strip()
            # remove white space in list
            percentile_list = num_params.split(' ')
            [x.strip() for x in percentile_list]
            # remove non-numerical values
            percentile_list_floats = []
            for percentile in percentile_list:
                try:
                    float(percentile)
                    percentile_list_floats.append(float(percentile))
                except:
                    percentile_list.remove(percentile)
            # check in ascending
            if sorted(percentile_list_floats) != percentile_list_floats:
                alert = not_ascending
                is_valid = False
            parsed_dist_info.append([dist[0], percentile_list_floats])
        else:
            parsed_dist_info.append(dist)
    return parsed_dist_info, is_valid, alert


def create_corr_data_set(df_vals, rho, desired_mean, desired_std, noise):
    is_valid = True
    alert = no_update
    sample_size = len(df_vals)
    try:
        dummy_vals, b, m = get_lin_params(df_vals, sample_size)
    except LinAlgError:
        print('the problem occured again...')
        alert = unknown_error
        is_valid = False
        return None, None, None, alert, is_valid
    residuals = get_resid(df_vals, b, m)

    std_data, std_resids = create_std_data(df_vals, residuals)

    corr_data = create_corr_data(df_vals, rho, noise)

    rho_sample, sp_rank = make_rho_sample(corr_data, df_vals)

    scaled_data = scale_data(corr_data, desired_mean, desired_std)
    print('x_vals...', df_vals)
    print('y_vals...', scaled_data)
    return scaled_data, rho_sample, sp_rank, alert, is_valid


def get_lin_params(vals, size):
    data_ints = np.asarray(vals)
    data_ints.astype(int)
    x = np.arange(1, size + 1, 1)
    b, m = polyfit(vals, x, 1)
    return x, b, m


def get_resid(vals, b, m):
    resids = []
    for i, val in enumerate(vals):
        resids.append(val - (m * i + b))
    return resids


def create_std_data(vals, resids):
    return np.std(vals), np.std(resids)


def create_corr_data(y, rho, noise):
    dummy_X1 = np.random.uniform(0, 1, len(y))
    if noise is not 0:
        noise_X3 = np.random.uniform(0, 1, len(y))
        noise_X3 = noise_X3 * noise
        dummy_X1 = dummy_X1 + noise_X3
    X2 = (y - y.mean())/y.std()
    data = rho * X2 + np.sqrt(1 - rho**2) * dummy_X1
    return data


def make_rho_sample(x, y):
    return np.corrcoef(x, y), spearmanr(x, y)


def scale_data(corr_data, desired_mean, desired_std):
    old_mean = np.mean(corr_data)
    old_std = np.std(corr_data)
    scaled_data = []
    for data_point in corr_data:
        new_point = desired_mean + (data_point - old_mean) * desired_std / old_std
        scaled_data.append(new_point)
    return scaled_data


def update_dict_with_assoc_num_list(old_dicts, new_dicts, num_assoc, cat_f_assoc, new_f_name):
    list_new_dfs = []
    for dist_dict in new_dicts:
        cat_assoc = dist_dict['cat-assoc']
        cat_assoc_list_cats = [cat_assoc] * dist_dict['freq']
        num_assoc_values = list(dist_dict['num-assoc'].values())
        new_values = list(dist_dict['data'].values())

        new_dist_dict = {dist_dict['feature-name']: new_values,
                         num_assoc: num_assoc_values,
                         dist_dict['feature-assoc']: cat_assoc_list_cats}
        num_assoc_df = pd.DataFrame(new_dist_dict)
        list_new_dfs.append(num_assoc_df)

    # join all dfs vertically
    concat_df = pd.concat(list_new_dfs, ignore_index=True)
    # change the indices to strings
    concat_df.index = [str(x) for x in range(1, len(concat_df)+1)]
    concat_dict = concat_df.to_dict()
    # now replace ALL used features with the ones from the DF
    old_dicts[new_f_name] = concat_dict[new_f_name]
    old_dicts[num_assoc] = concat_dict[num_assoc]
    if cat_f_assoc != 'no':
        old_dicts[cat_f_assoc] = concat_dict[cat_f_assoc]
    return old_dicts


def update_num_list_with_assoc(orig_dict, num_f_assoc, cat_f_assoc, new_f_name, list_of_lists):
    orig_df = pd.DataFrame(orig_dict)
    new_dfs = []
    is_num_f_assoc = False if 'no' in num_f_assoc else True
    is_cat_f_assoc = False if 'no' in cat_f_assoc else True
    for new_dict in list_of_lists:
        if is_cat_f_assoc:
            assoc_cat = new_dict['cat-assoc']
            # # for debug
            # orig_dict = orig_df.to_dict()
            df_filt = orig_df[orig_df[cat_f_assoc] == assoc_cat]
            # if is_num_f_assoc:
            #     df_filt = df_filt.sort_values(by=[num_f_assoc])
            # new_series = pd.Series(new_dict['data']).sort_values()
            new_series = pd.Series(new_dict['data'])
            new_series.index = df_filt.index
            df_filt[new_f_name] = new_series
            new_dfs.append(df_filt)
        else:
            # orig_df = orig_df.sort_values(by=[num_f_assoc])
            # new_series = pd.Series(new_dict['data']).sort_values()
            new_series = pd.Series(new_dict['data'])
            new_series.index = orig_df.index
            orig_df[new_f_name] = new_series
    df = pd.concat(new_dfs, ignore_index=True) if len(list_of_lists) > 1 else orig_df
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def get_num_title(num_assoc, cat_assoc, is_corr):
    title_string_corr = 'Generate data which correlates with ' + num_assoc + '...'
    title_string_corr += ' split by ...' if cat_assoc is not 'no' else ''
    title_string_dist = 'Generate a data distribution'
    title_string_dist += ' split by ...' if cat_assoc is not 'no' else ''
    if is_corr:
        return title_string_corr
    else:
        return title_string_dist
