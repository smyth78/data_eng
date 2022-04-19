import json
import pandas as pd

import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc




def get_current_df_as_table(cols, rows):
    table = dash_table.DataTable(
        id='de-current-df-as-table',
        columns=cols,
        data=rows,
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


def get_created_data_as_table(cols, rows):
    table = dash_table.DataTable(
        # id={'type': 'de-created-data-as-table', 'index': 0},
        columns=cols,
        data=rows,
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


def creation_check_box_div(df_already_exists):
    div = html.Div([dbc.Checkbox(
        id={'type': "de-hide-creation-check", 'index': 0}, className="form-check-input", checked=True if df_already_exists else False
        ),
        dbc.Label(
            "Hide spreadsheet creation section",
            html_for="de-hide-creation-div",
            className="form-check-label",
        ),
    ])
    return div


def split_dfs_by_data_type_from_stored_data(df_as_list):
    df = pd.DataFrame.from_dict(df_as_list, orient='columns')

    # get numerical data
    df_num = df.select_dtypes(include=['number'])

    # find the cols not in numerical data
    cat_cols = [c for c in df.columns if c not in df_num.columns]

    # define categorical DF
    df_cat = df[cat_cols]
    return df_num, df_cat


def extract_json_from_dynamic_callback(cb_string):
    cb_list = cb_string.split('.')
    try:
        element_dict = json.loads(cb_list[0])
    except:
        print('not a json')
        element_dict = None
    return element_dict

