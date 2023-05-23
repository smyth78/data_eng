import numpy as np
import pandas as pd
import decimal

from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash import no_update
import plotly.express as px

from helper_modules.alerts import *
from helper_modules.constants import *

def verify_valid_new_df(name, size, data_dict):
    is_new = True
    alert = None
    if data_dict['name'] == name and data_dict['size'] == size:
        is_new = False
        alert = not_new_df
    try:
        int(size)
    except ValueError:
        print('size is not int')
        alert = must_be_integer

    return is_new, alert


def get_df_title_div(stored_df_dict):
    if stored_df_dict['name'] is not None and stored_df_dict['size'] is not None:
        div = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H3(stored_df_dict['name'], style={'color': 'red'}),
                    html.H5('Rows: ' + str(stored_df_dict['size'])),
                    html.Div('Current spreadsheet'),
                ])
            ])
        ])
    else:
        div = html.Div(dbc.Row(dbc.Col(html.H5('Define new spreadsheet to continue...'))))
    return div


def is_all_values_of_dict_not_none(dictionary):
    return not all(dictionary.values())


def check_valid_entry(tables_list_dicts, size, prob_type, df, feat_assoc, cat_assoc_list):
    is_valid = True
    alert = no_update

    total_type = 'probability' if 'probability' in prob_type else 'frequency'

    # if it has assoc feature then must word out respective frequencies and put in size list
    if any('no' not in d for d in tables_list_dicts):
        df = pd.DataFrame.from_dict(df)
        dff = df.groupby(feat_assoc[0]).size().reset_index(name='freq').rename(columns={'Col1': 'Col_value'})
        dff.reset_index(drop=True, inplace=True)

        size = []
        for cat in cat_assoc_list:
            df_chosen_row = dff.loc[dff[feat_assoc[0]] == cat]
            size.append([cat, int(df_chosen_row['freq'])])
    else:
        size = [['no', size]]

    for data_defn in size:
        assoc_feat = data_defn[0]
        cat_size = data_defn[1]
        table_total_frequency = 0
        table_list_of_floats_as_strings = []
        for list_dicts in tables_list_dicts[assoc_feat]:
            # first check to see if any entries are empty
            if is_all_values_of_dict_not_none(list_dicts):
                is_valid = False
                alert = some_values_none
                return None, alert, is_valid
            if total_type == 'probability':
                # test all entries in prob are floats
                try:
                    table_list_of_floats_as_strings.append(str(list_dicts['prob/freq']))
                except ValueError:
                    is_valid = False
                    alert = val_error
                    return None, alert, is_valid
            else:
                try:
                    table_total_frequency += int(list_dicts['prob/freq'])
                except ValueError:
                    is_valid = False
                    alert = val_freq_error
                    return None, alert, is_valid

        if total_type == 'probability':
            if str(sum(map(decimal.Decimal, table_list_of_floats_as_strings ))) != str(1.0):
                is_valid = False
                alert = prob_error
                return None, alert, is_valid
        else:
            if table_total_frequency != cat_size:
                is_valid = False
                # decide if the error is for a assoc feature or not
                alert = freq_error_single if len(size) == 1 else freq_error_assoc
                return None, alert, is_valid

    return tables_list_dicts, alert, is_valid


def make_one_way_freq_table_from_df(df, col_name):
    dff = df.groupby(col_name).size().reset_index(name='Freq').rename(columns={'Col1': 'Col_value'})
    dff.reset_index(drop=True, inplace=True)
    column_names = [{"name": i, "id": i} for i in dff.columns]
    freq_df_dict = dff.to_dict('records')
    data_f = freq_df_dict
    return column_names, data_f


def make_dash_table(columns_f, data_f, index):
    table = dash_table.DataTable(
        id={'type': 'cat-freq-display', 'index': index},
        columns=columns_f,
        data=data_f,
        editable=False,
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'width': '100px',
            'maxWidth': '100px',
            'minWidth': '100px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        export_format="csv",
        fill_width=False,
        page_size=10,
    )
    return table


def get_freq_table_from_dataframe(df, feature_name):
    dff = df.groupby(feature_name).size().reset_index(name='Frequency').rename(columns={'Col1': 'Col_value'})
    dff.reset_index(drop=True, inplace=True)
    column_names = [feature_name, 'Frequency']
    return dff, column_names


def get_distinct_cats_from_cat_feature(df, feature_name):
    dff = df.groupby(feature_name).size().reset_index(name='Frequency').rename(columns={'Col1': 'Col_value'})
    dff.reset_index(drop=True, inplace=True)
    cat_list = dff[feature_name].values.tolist()
    cat_list = set(cat_list)
    return cat_list


def make_cat_main_chart_output_div(created_df, feature_name, is_assoc_feat, y_names, dash_table):
    # figure_labels = {'x': feature_name, 'y': 'Frequency'}
    # create bar figure
    bar_chart = px.histogram(created_df, x=feature_name, y=y_names, barmode='group', template="simple_white") if \
        is_assoc_feat else px.histogram(created_df, x=feature_name, template="simple_white")
    # is_assoc_feat else px.bar(created_df[feature_name], template="simple_white")
    bar_chart.update_layout(xaxis_title=feature_name, yaxis_title='Frequency', legend_title="",
                            showlegend=True if is_assoc_feat else False, margin=dict(l=0, r=0, t=30, b=0))
    bar_fig = dcc.Graph(figure=bar_chart, id={'type': 'cat-bar-display', 'index': 0}, className="w-100")
    div = dbc.Col([dbc.Row(html.H5('Summary of data created...')),
                   dbc.Row([dash_table]),
                   dbc.Row([bar_fig])
                   ],
                  width=6, style={'margin-right': COLUMN_PAD}, id={'type': 'cat-div-display', 'index': 0})

    return div


def make_cat_data(table_data, size, prob_type, is_assoc, assoc_feature):
    # if prob then sample, if freq then exact.
    # note: the probs AND freq vals have already been verified

    parse_tables = []
    prob_type = 'probability' if 'probability' in prob_type else 'frequency'
    for key, size in zip(table_data, size):
        parse_cat = []
        for row in table_data[key]:
            parse_cat.append([row['category'], row['prob/freq']])
        parse_tables.append([key, parse_cat, size])

    if prob_type == 'frequency':
        cat_data = []
        for table in parse_tables:
            assoc_cat = table[0]
            parsed_cat =[]
            for cat in table[1]:
                for i in range(int(cat[1])):
                    parsed_cat.append(cat[0])
            np.random.shuffle(parsed_cat)
            cat_data.append([assoc_cat, parsed_cat])
    else:
        # this is for prob
        cat_data = []
        for table in parse_tables:
            assoc_cat = table[0]
            sample_list = []
            prob_list = []
            for cat in table[1]:
                sample_list.append(cat[0])
                prob_list.append(float(cat[1]))
            sample_size = size[1] if not is_assoc else table[2][1]
            table_data = np.random.choice(sample_list, int(sample_size), p=prob_list)
            np.random.shuffle(table_data)
            cat_data.append([assoc_cat, table_data])
    return cat_data


def create_cat_table(index):
    div = html.Div([
        dcc.Store(id={'type': 'de-cat-store', 'index': index}),
        html.Div(id={'type': 'de-alert-cat', 'index': index}),
        dash_table.DataTable(
            id={'type': 'de-cat-table', 'index': index},
            editable=True,
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'width': '100px',
                'maxWidth': '100px',
                'minWidth': '100px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        )
    ])
    return div


def categorical_creation_table(index, df_cat, cat_assoc):
    is_cat_assoc = True if cat_assoc is not 'no' else False
    # categories is either the assoc cats or a single empty string to represent a new feature
    categories = get_distinct_cats_from_cat_feature(df_cat, cat_assoc) if is_cat_assoc else ['']
    element_row = []
    for cat in categories:
        frequency_of_cat = df_cat[cat_assoc].value_counts()[cat].sum() if is_cat_assoc else None
        cat_string = str(cat) + ' (' + str(frequency_of_cat) + ')' if is_cat_assoc else None
        div = dbc.Col([
                    dbc.Row('...new feature related to ' + str(cat_string)) if is_cat_assoc else None,
                    dcc.Store(id={'type': 'de-cat-assoc-name', 'index': index}, data=cat if is_cat_assoc else 'no'),
                    dbc.Row([
                        dbc.Input(id={'type': 'de-category-number', 'index': index},
                                  type="number",
                                  min=1, max=20, step=1,
                                  placeholder='No. of categories...',
                                  style={'display': 'inline-block'})
                    ], style={'paddingBottom': ROW_MARGIN}),
                    dbc.Row([
                        dbc.Col([
                            create_cat_table(index)
                        ])
                    ]),
                ], width=3, style={'marginRight': COLUMN_PAD})
        index += 1
        element_row.append(div)
    return element_row, index

def get_dynamic_radio(df_cat, index, radio_choice):
    options = [{"label": i, "value": i} for i in df_cat.columns] if df_cat is not None else []
    options.append({'label': 'No', 'value': 'no'})
    div = html.Div([
        dbc.Row([html.H5('Associate new feature with existing categorical feature...')]),
        dbc.RadioItems(
            id={'type': 'de-cat-assoc', 'index': index},
            inline=True,
            options=options,
            value=radio_choice,
        )
    ])
    return div


def cat_creation_div(index, cat_assoc, df_cat, current_radio_choice):
    element_row, index = categorical_creation_table(index, df_cat, cat_assoc)
    div = html.Div([
        dbc.Row([get_dynamic_radio(df_cat, index, current_radio_choice)], id={'type': 'de-cat-assoc-div', 'index': index}),
        THEME_BREAK,
        dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.RadioItems(
                        id={'type': 'de-category-prob-choice', 'index': index},
                        inline=True,
                        options=[
                            {'label': 'Probability', 'value': 'probability'},
                            {'label': 'Frequency', 'value': 'frequency'},
                        ],
                        value='probability',
                        style={'margin': '0 0px 10px'})]),
                dbc.Row(element_row, id='main-chart-holder')
            ], id='de-category-table-div'),
    ])
    ]),
    ])
    return div, index


def update_dict_with_assoc_cat_list(old_dict, new_dicts):
    updated_df = pd.DataFrame.from_dict(old_dict)
    new_feature_name = new_dicts[0]['feature-name']
    assoc_feature_name = new_dicts[0]['feature-assoc']

    # make 2-D lists of the assoc_cat and the new_cat
    cat_lists = []
    for new_cat_dict in new_dicts:
        for cat_value in new_cat_dict['data'].values():
            cat_lists.append([new_cat_dict['cat-assoc'], cat_value])
    updated_df = updated_df.sort_values(assoc_feature_name)
    sorted_cat_and_assoc_list = sorted(cat_lists, key=lambda x: x[0])
    sorted_just_cat = [row[1] for row in sorted_cat_and_assoc_list]
    updated_df[new_feature_name] = sorted_just_cat
    updated_df.sort_index(inplace=True)
    updated_dict = updated_df.to_dict()
    return updated_dict


def update_size_depending_on_assocs(df_cat, cat_assoc, df_full_size):
    is_cat_assoc = True if cat_assoc is not None else False
    cat_freqs_data = []
    if is_cat_assoc:
        # categories is either the assoc cats or a single empty string to represent a new feature
        categories = get_distinct_cats_from_cat_feature(df_cat, cat_assoc) if is_cat_assoc else ['']
        for cat in categories:
            frequency_of_cat = df_cat[cat_assoc].value_counts()[cat].sum() if is_cat_assoc else None
            cat_freqs_data.append([cat, frequency_of_cat])
    else:
        cat_freqs_data.append(['no', df_full_size])
    return cat_freqs_data


def split_cat_data_by_sec_f(df, pri_feat, sec_feat):
    pri_cats = np.unique(df[pri_feat].values)
    sec_cats = np.unique(df[sec_feat].values)
    column_titles = list(pri_cats)
    column_titles.insert(0, sec_feat)
    list_for_df_freq_table = [column_titles]
    column_names = []
    for sec_cat in sec_cats:
        column_names.append(sec_cat)
        # make an empty array as long as the primary features
        sec_feature_data = [sec_cat]
        chosen_features = df[[pri_feat, sec_feat]]
        for pri_cat in pri_cats:
            filtered_df = chosen_features[(chosen_features[pri_feat] == pri_cat)]
            # get the row count
            count_df = filtered_df[filtered_df[sec_feat] == sec_cat].shape[0]
            sec_feature_data.append(count_df)

        list_for_df_freq_table.append(sec_feature_data)
    df = pd.DataFrame(list_for_df_freq_table)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    y_names = list(pri_cats)
    return df, y_names, list_for_df_freq_table, column_names


def make_two_way_table(list_or_df_freq_table):
    # format the table appropriatly
    df = pd.DataFrame(list_or_df_freq_table)
    df.columns = df.iloc[0]
    df = df[1:]

    # add the correct column/row titles
    sec_feat_titles = []
    for sec_feat_title in list_or_df_freq_table:
        sec_feat_titles.append(sec_feat_title[0])
    # df.insert(0, sec_feat_titles[0], sec_feat_titles[1:])

    freq_df_dict = df.to_dict('records')
    columns_f = [{"name": i, "id": i} for i in df.columns]
    data_f = freq_df_dict
    return columns_f, data_f