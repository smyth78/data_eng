import json
import pandas as pd

import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State, MATCH, ALL
from dash import no_update
import dash._callback_context as cb_ctx

from server import app, server

from helper_modules.constants import *
from helper_modules.categorical_helpers import *
from helper_modules.numerical_helpers import *
from helper_modules.time_helpers import *
from helper_modules.general_helpers import *
from helper_modules.alerts import *

def layout():
    layout = html.Div([
        dcc.Store(id='de-index'),
        dcc.Store(id='de-index-display-data', data=0),
        dcc.Store(id='de-temp-data-dict-store'),
        dcc.Store(id='de-current-df-store', storage_type='session'),
        dcc.ConfirmDialog(
            id='de-warn-new-df',
            message='Be careful...this will erase any existing data...',
        ),
        html.Div([], id='de-alert-1'),
        dbc.Button(id="de-dummy-new-df-confirm", n_clicks=0, style={'display': 'none'}),
        # hide ss creation checkboxes
        dbc.Row([
            dbc.Col([
                dbc.Row(id='de-current-df-title-div', style={'paddingBottom': ROW_MARGIN, 'paddingTop': ROW_MARGIN}),
                dbc.Row([
                    dbc.FormGroup(
                        [],
                        id='de-hide-creation-div',
                        check=True,
                    )
                ]),
                dbc.Row([
                    dbc.FormGroup(
                        [
                            dbc.Checkbox(
                                id="de-hide-table-div", className="form-check-input"
                            ),
                            dbc.Label(
                                "Show spreadsheet data",
                                html_for="de-hide-table-div",
                                className="form-check-label",
                            ),
                        ],
                        check=True,
                    )
                ])
            ], width=4),
            dbc.Col([
                dash_table.DataTable(
                    id='de-current-df-as-table',
                    columns=None,
                    data=None,
                    editable=False,
                )
            ], id='de-current-df-as-table-div', width=8)
        ]),
        # create new ss
        html.Div([
            dbc.Row([dbc.Col([
                dbc.Row([
                    dbc.Label('Spreadsheet name'),
                    dbc.Input(id='de-df-name',
                              type='text',
                              placeholder='Enter name...')]),
                dbc.Row([
                    dbc.Label('No. of rows in spreadsheet'),
                    dbc.Input(id='de-df-size',
                              type='number',
                              placeholder='Enter size...',
                              min=1, max=2000, step=1)], style={'paddingBottom': ROW_MARGIN}),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("New", color="danger", className="mr-1", id='de-create-new-df',
                                   n_clicks=0)
                    ]),
                ]),
            ], width=4),
        ]),
            ], id='de-df-creation-div'),
        THEME_BREAK,
        # create new feature and name it
        dbc.Row([dbc.Col([
            dbc.Row([html.H5('Create new feature...')]),
            dbc.Row([
                dbc.RadioItems(
                    id='de-data-choice',
                    inline=True,
                    options=[
                        {'label': 'Categorical', 'value': 'categorical'},
                        {'label': 'Numerical', 'value': 'numerical'},
                        {'label': 'Time series', 'value': 'time'},
                    ],
                    value='categorical',
                    style={'margin': '0 0px 10px'}
                ),
            ]),
            dbc.Row([dbc.Input(id='de-feature-name',
                               type='text',
                               placeholder='New feature name...')]),
            dbc.Row(html.Div(id='de-create-data-alert'), style={'paddingTop': '10px'}),
            THEME_BREAK,
        ], width=4)]),
        # this is the div that will be customised when radio buttons pressed
        html.Div([], id='de-main-div-alert'),
        html.Div([], id='de-main-div'),
        THEME_BREAK,
        # this is the chart that will display the created data
        html.Div(id='de-main-chart-alert'),
        dbc.Col(id='de-main-chart'),
        THEME_BREAK,
        # create and append buttons
        dbc.Row([dbc.Col([
                dbc.Row([
                    dbc.Col(
                        [dbc.Button("Create", color="success", className="mr-1", id='de-create-data', n_clicks=0)],
                        width=3),
                    dbc.Col(
                        [dbc.Button("Append", color="warning", className="mr-1", id='de-append-data', n_clicks=0)],
                        width=3),
                ], justify="between"),
            ], width=4)])
        ], style={'padding-bottom': PAGE_BOTTOM_PAD})
    return layout


# retrive and store DF plus create new....only triggered once diaglog is confirmed
@app.callback(
        [Output('de-current-df-store', 'data'),
         Output('de-current-df-title-div', 'children'),
         Output('de-df-creation-div', 'style'),
         Output('de-current-df-as-table-div', 'style'),
         Output('de-alert-1', 'children'),
         Output('de-current-df-as-table-div', 'children'),
         Output('de-hide-creation-div', 'children'),
         Output('de-main-chart-alert', 'children')],
         [Input('de-append-data', 'n_clicks'),
          Input('de-warn-new-df', 'submit_n_clicks'),
          Input({'type': 'de-hide-creation-check', 'index': ALL}, 'checked'),
          Input('de-hide-table-div', 'checked')],
        [State('de-current-df-store', 'data'),
         State('de-df-name', 'value'),
         State('de-df-size', 'value'),
         State('de-temp-data-dict-store', 'data')]
)
def data_type_selector(append_click, warn_click, is_hide_creation_div, is_hide_table_div, stored_df_dict, name, size,
                       updated_pop_dict):
    ctx = cb_ctx
    main_alert = no_update
    chart_alert = no_update

    default_stored_dict = {'name': None, 'size': None, 'data': None}

    new_num_assoc_options, new_cat_assoc_options = [no_update], [no_update]

    is_hide_creation_div = False if is_hide_creation_div == [] or False in is_hide_creation_div else True
    is_hide_table_div = False if is_hide_table_div is None or is_hide_table_div == False else True

    table_ss_div_style = {'display': 'none' if not is_hide_table_div else 'block'}

    df_exists = False if stored_df_dict is None or stored_df_dict == default_stored_dict else True
    df_creation_div_style = {'display': 'none' if df_exists else 'block'}

    # if df_exists hide creation div otherwise dont hide it
    hide_creation_checkbox = creation_check_box_div(df_exists)

    # if the checkbox was clicked then do as the check says!
    element_dict = extract_json_from_dynamic_callback(ctx.callback_context.triggered[0]['prop_id'])
    if element_dict is not None:
        if element_dict['type'] == 'de-hide-creation-check':
            hide_creation_checkbox = creation_check_box_div(is_hide_creation_div)
            df_creation_div_style = {'display': 'none' if is_hide_creation_div else 'block'}

    # start the stored dict
    stored_df_dict = stored_df_dict if df_exists else {'name': None, 'size': None, 'data': {}}

    # which button was clicked?
    if ctx.callback_context.triggered[0]['prop_id'] == 'de-append-data.n_clicks':
        if updated_pop_dict is not None:
            stored_df_dict['data'] = updated_pop_dict
            chart_alert = data_successfully_appended
        else:
            chart_alert = problem_appending_data
    # a new callback is triggered to create the new_df once confirmed
    elif ctx.callback_context.triggered[0]['prop_id'] == 'de-warn-new-df.submit_n_clicks':
        is_valid_new_df, main_alert = verify_valid_new_df(name, size, stored_df_dict)
        hide_creation_checkbox = creation_check_box_div(True)
        stored_df_dict = {'name': name, 'size': size, 'data': {}} if is_valid_new_df else stored_df_dict
        df_creation_div_style = {'display': 'none'}
        new_num_assoc_options = {'label': 'No', 'value': 'no'}
        new_cat_assoc_options = {'label': 'No', 'value': 'no'}

    current_df_title_div = get_df_title_div(stored_df_dict)
    current_df = pd.DataFrame.from_dict(stored_df_dict['data'])
    current_df_as_table = get_current_df_as_table([{"name": i, "id": i} for i in current_df.columns],
                                                  current_df.to_dict('records'))
    return stored_df_dict, current_df_title_div, df_creation_div_style, table_ss_div_style, main_alert, current_df_as_table, \
           hide_creation_checkbox, chart_alert

# this is called when the confirm button from the dialog is pressed
@app.callback(
    Output("de-dummy-new-df-confirm", 'n_clicks'),
    [Input('de-warn-new-df', 'submit_n_clicks'),
     Input('de-warn-new-df', 'cancel_n_clicks')],
    [State('de-current-df-store', 'data')]
)
def update_output(submit_n_clicks, cancel_n_clicks, data_dict):
    return_value = 0
    ctx = cb_ctx
    if submit_n_clicks is not None or cancel_n_clicks is not None:
        if submit_n_clicks > 0 or cancel_n_clicks > 0:
            if ctx.callback_context.triggered[0]['prop_id'] == 'de-warn-new-df.submit_n_clicks':
                return_value = 1
            elif ctx.callback_context.triggered[0]['prop_id'] == 'de-warn-new-df.cancel_n_clicks':
                return_value = 0

    return return_value


# this callback displays the dialog box when NEW is clicked
@app.callback(
    Output('de-warn-new-df', 'displayed'),
    Input('de-create-new-df', 'n_clicks')
)
def update_output(new_click):
    return True if new_click > 0 else False


# radio button data type selector - CHANGE THE DIVS based on data type choice
@app.callback(
        [Output('de-main-div', 'children'),
         Output('de-index', 'data'),
         Output('de-main-div-alert', 'children'),
         Output('de-feature-name', 'value')],
        [Input('de-data-choice', 'value'),
         Input('de-current-df-store', 'data'),
         Input('de-warn-new-df', 'submit_n_clicks'),
         Input({'type': 'de-cat-assoc', 'index': ALL}, 'value'),
         Input({'type': 'de-num-assoc', 'index': ALL}, 'value'),
         Input("de-dummy-new-df-confirm", 'n_clicks'),
         Input({'type': 'de-time-func', 'index': ALL}, 'value')
         ],
        [State('de-index', 'data'),
         State('de-current-df-store', 'data'),
         State('de-feature-name', 'data')]
)
def data_type_selector(feature_data_type, data_store, new_click, cat_assoc, num_assoc, dialog_click, times_func,
                       index, current_df, feature_name):
    ctx = cb_ctx
    alert = no_update
    dff_as_list = current_df.copy()
    main_div = no_update
    index = 0 if index is None else index

    # output_chart_style = [{'display': 'none' if 'no' in num_assoc else 'block'}]

    if ctx.callback_context.triggered[0]['prop_id'] == 'de-warn-new-df.submit_n_clicks':
        is_new = True
    else:
        is_new = False

    feature_name = feature_name if not is_new else ''

    if feature_data_type == 'categorical':
        if dff_as_list['data'] is not None:
            _, df_cat = split_dfs_by_data_type_from_stored_data(dff_as_list['data'])
        else:
            df_cat = None
        # this next line is set assoc cat to None iff
        cat_assoc = 'no' if (cat_assoc == []) or ('no' in cat_assoc) or is_new else cat_assoc[0]
        main_div, index = cat_creation_div(index, cat_assoc, df_cat, cat_assoc)
        index += 1
    elif feature_data_type == 'numerical':
        if dff_as_list['data'] is not None:
            df_num, df_cat = split_dfs_by_data_type_from_stored_data(dff_as_list['data'])
        else:
            df_num, df_cat = None, None
        # need to define these inputs as they dont exist on first callbakc
        default_num_dist_type = ['normal']
        cat_assoc = 'no' if (cat_assoc == []) or ('no' in cat_assoc) or is_new else cat_assoc[0]
        num_assoc = 'no' if (num_assoc == []) or ('no' in num_assoc) or is_new else num_assoc[0]
        main_div, index, alert = num_creation_div(index, df_num, df_cat, default_num_dist_type, cat_assoc, num_assoc)
    elif feature_data_type == 'time':
        main_div, index, alert = time_creation_div(index, times_func)

    return main_div, index, alert, feature_name


# this is to change the paras of the num div when the dist type is changed dynamiclaly
@app.callback(
        Output({'type': 'de-num-para-input', 'index': MATCH}, 'children'),
        [Input({'type': 'de-num-dist-type', 'index': MATCH}, 'value')],
)
def data_type_selector(num_dist_type):
    ctx = cb_ctx
    params = None
    if num_dist_type == 'normal':
        params = normal_params()
    elif num_dist_type == 'uniform':
        params = uniform_params()
    elif num_dist_type == 'exponential':
        params = exponential_params()
    elif num_dist_type == 'skew':
        params = skew_params()
    elif num_dist_type == 'percentiles':
        params = percentiles_params()
    return params


# change the freq/prob in category table - MATCH means will only fire when it exists
@app.callback(
        [Output({'type': 'de-cat-table', 'index': MATCH}, 'columns'),
         Output({'type': 'de-cat-table', 'index': MATCH}, 'data')],
        [Input({'type': 'de-category-number', 'index': MATCH}, 'value')],
)
def data_type_selector(cat_number):
    ctx = cb_ctx
    cat_number = cat_number if cat_number is not None else 0
    # data = [dict(category=np.nan, probfreq=np.nan) for i in range(0, cat_number)]
    data = [{'category': np.nan, 'prob/freq': np.nan} for i in range(0, cat_number)]
    columns = [{'id': 'category', 'name': 'category'},
               {'id': 'prob/freq', 'name': 'prob/freq'}]
    return columns, data


# Create data, display charts/data in main chart div and then store data awaiting appendment
@app.callback(
         [Output('de-create-data-alert', 'children'),
          Output('de-temp-data-dict-store', 'data'),
          Output('de-main-chart', 'children'),
          Output('de-index-display-data', 'data')],
        [Input('de-create-data', 'n_clicks'),
         Input('de-data-choice', 'value'),
         Input({'type': 'de-cat-assoc', 'index': ALL}, 'value'),
         Input({'type': 'de-num-assoc', 'index': ALL}, 'value')],
        [State({'type': 'de-cat-assoc-name', 'index': ALL}, 'data'),
         State({'type': 'de-num-assoc-type', 'index': ALL}, 'value'),
         State({'type': 'de-cat-table', 'index': ALL}, 'data'),
         State('de-current-df-store', 'data'),
         State({'type': 'de-category-prob-choice', 'index': ALL}, 'value'),
         State('de-feature-name', 'value'),
         State('de-data-choice', 'value'),
         State('de-index-display-data', 'data'),
         State({'type': 'de-num-input-div', 'index': ALL}, 'children'),
         State({'type': 'de-num-chart-combo', 'index': ALL}, 'value')]
)
def data_type_selector(create_click, change_new_feature_type, cat_assoc_radio, num_assoc_radio, distinct_cats_assoc, num_assoc_type, cat_table_data, data_dict,
                       cat_prob_type, new_f_name, data_type, index_display_data, num_input_div, output_chart):
    ctx = cb_ctx
    alert = None
    is_valid_dict = True
    created_data_list_of_dicts = None
    updated_pop_dict = no_update

    name = data_dict['name'] if data_dict is not None else None
    size = data_dict['size'] if data_dict is not None else None

    feature_assoc = None if 'no' in cat_assoc_radio or cat_assoc_radio == [] else cat_assoc_radio[0]
    distinct_cats_assoc = [None] if distinct_cats_assoc == [] else distinct_cats_assoc

    is_assoc_feature = False if feature_assoc is None else True

    parsed_cat_table_data_dict = {}
    for table, related_cat in zip(cat_table_data, distinct_cats_assoc):
        if table is not None:
            parsed_cat_table_data_dict[related_cat] = table

    main_chart = []
    # this clears the main chart when the data type is changed
    if ctx.callback_context.triggered[0]['prop_id'] == 'de-data-choice.value':
        return alert, created_data_list_of_dicts, dbc.Row(main_chart), index_display_data

    elif ctx.callback_context.triggered[0]['prop_id'] == 'de-create-data.n_clicks':
        # return a warning if feature is not named or duplicated
        if new_f_name is None or new_f_name == "":
            return must_name_feature, no_update, no_update, no_update
        if data_dict['data'] is not None:
            if new_f_name in list(data_dict['data'].keys()):
                return duplicate_feature_name, no_update, no_update, no_update
        if data_type == 'categorical':
            # this check is to verify the cat input boxes are valid inputs
            cat_table_data, alert, is_valid = check_valid_entry(parsed_cat_table_data_dict, size, cat_prob_type,
                                                                data_dict['data'], cat_assoc_radio, distinct_cats_assoc)
            if not is_valid:
                # if the inputted table data is incorrect then an alert is sent immdeiaely
                return alert, no_update, no_update, no_update

            _, df_cat = split_dfs_by_data_type_from_stored_data(data_dict['data'])
            size_list = update_size_depending_on_assocs(df_cat, feature_assoc, size)
            cat_data_array_of_arrays = make_cat_data(cat_table_data, size_list, cat_prob_type, is_assoc_feature, cat_assoc_radio[0])
            created_data_list_of_dicts = []
            for cat_data_array in cat_data_array_of_arrays:
                created_data_list_of_dicts.append({'feature-name': new_f_name,
                                               'feature-assoc': feature_assoc,
                                               'cat-assoc': cat_data_array[0],
                                               'freq': len(list(cat_data_array[1])) if
                                               'probability' in cat_prob_type else len(cat_data_array[1]),
                                               'data': dict(enumerate(cat_data_array[1].flatten(), 1)) if
                                               'probability' in cat_prob_type else dict(enumerate(cat_data_array[1]))
                                               })
            # make combined df of stored data and created data
            if is_assoc_feature:
                updated_pop_dict = update_dict_with_assoc_cat_list(data_dict['data'], created_data_list_of_dicts)
                updated_df = pd.DataFrame.from_dict(updated_pop_dict)
                updated_df.index = [str(x) for x in range(1, len(updated_df) + 1)]
                updated_pop_dict = updated_df.to_dict()
                updated_df, y_names, list_for_df_freq_table, column_names = split_cat_data_by_sec_f(updated_df,
                                                                                                    feature_assoc,
                                                                                                    new_f_name)
                columns_f, data_f = make_two_way_table(list_for_df_freq_table)
            else:
                new_dict = created_data_list_of_dicts[0]
                data_dict['data'][new_dict['feature-name']] = dict(new_dict['data'])
                updated_pop_dict = data_dict['data']
                updated_df = pd.DataFrame.from_dict(updated_pop_dict)
                updated_df.index = [str(x) for x in range(1, len(updated_df) + 1)]
                updated_pop_dict = updated_df.to_dict()
                columns_f, data_f = make_one_way_freq_table_from_df(updated_df, new_dict['feature-name'])
            dash_freq_table = make_dash_table(columns_f, data_f, index_display_data)
            index_display_data += 1
            main_chart.append(make_cat_main_chart_output_div(updated_df, new_f_name, is_assoc_feature,
                                                             distinct_cats_assoc, dash_freq_table))
        elif data_type == 'numerical':
            is_cat_f_assoc = False if 'no' in cat_assoc_radio else True
            is_num_f_assoc = False if 'no' in num_assoc_radio else True
            num_assoc = 'no' if is_num_f_assoc is False else num_assoc_radio[0]
            cat_assoc = 'no' if is_cat_f_assoc is False else cat_assoc_radio[0]

            # find the input div
            num_input_children = None
            alert = no_update
            for key in ctx.callback_context.states:
                if is_dynamic_input_num_div(key):
                    num_input_children = ctx.callback_context.states[key]

            # parse deg acc and type of number
            is_dp, deg_of_acc, is_float, is_non_negative = parse_number_type(num_input_div, 1)

            # # this is no and cat
            # if not is_num_f_assoc:
            dist_info, is_valid, alert = get_dist_and_params(num_input_children) if not is_num_f_assoc \
                else get_corr_and_params(num_input_div, num_assoc)
            if not is_valid:
                return alert, created_data_list_of_dicts, dbc.Row(main_chart), index_display_data
            # parse percentile params here
            dist_info, is_valid, alert = parse_percentiles(dist_info) if not is_num_f_assoc else dist_info, is_valid, alert
            if not is_valid:
                return alert, created_data_list_of_dicts, dbc.Row(main_chart), index_display_data
            is_valid, alert = verify_num_dist_input_data(dist_info, is_num_f_assoc)
            if not is_valid:
                return alert, created_data_list_of_dicts, dbc.Row(main_chart), index_display_data
            df = pd.DataFrame.from_dict(data_dict['data'], orient='columns')
            df_num, df_cat = split_dfs_by_data_type_from_stored_data(data_dict['data'])
            size_list = update_size_depending_on_assocs(df_cat, feature_assoc, size)
            # parse param and cat data
            dist_info = add_sizes_too_dist_info(dist_info, size_list, is_cat_f_assoc, is_num_f_assoc)
            # generate data for none corr data
            data_lists, alert, is_valid = make_num_data_list(dist_info, is_num_f_assoc, num_assoc, cat_assoc, df)
            if not is_valid:
                return alert, created_data_list_of_dicts, dbc.Row(main_chart), index_display_data
            created_data_list_of_dicts = []
            # here we append the newly created data points
            for num_data in data_lists:
                rounded_list = round_list(is_dp, deg_of_acc, num_data, is_float, is_non_negative, is_num_f_assoc, None, False)
                new_data_dict = {str(i): j for i, j in enumerate(rounded_list, start=1)}
                feature_assoc = feature_assoc if feature_assoc is not None else 'no'
                if num_assoc is 'no':
                    if cat_assoc is 'no':
                        this_cat_assoc = 'no'
                    else:
                        this_cat_assoc = num_data[0]
                else:
                    this_cat_assoc = num_data[0][2]
                if is_num_f_assoc:
                    num_assoc_data = df.loc[df[feature_assoc] == this_cat_assoc, num_assoc].values if \
                        is_cat_f_assoc else df[num_assoc].values
                    num_assoc_dict = {str(i): j for i, j in enumerate(num_assoc_data, start=1)}
                else:
                    num_assoc_data = 'no'
                    num_assoc_dict = 'no'
                created_data_list_of_dicts.append({'feature-name': new_f_name,
                                                   'feature-assoc': feature_assoc,
                                                   'cat-assoc': this_cat_assoc,
                                                   'num-assoc': num_assoc_dict,
                                                   'freq': len(new_data_dict),
                                                   'data': new_data_dict
                                                  })
            # now update the existing data dict with the new data
            if is_cat_f_assoc or is_num_f_assoc:
                pop_df = update_num_list_with_assoc(data_dict['data'], num_assoc, cat_assoc, new_f_name,
                                                    created_data_list_of_dicts)
            else:
                data_dict['data'][created_data_list_of_dicts[0]['feature-name']] = created_data_list_of_dicts[0]['data']
                updated_pop_dict = data_dict['data']
                pop_df = pd.DataFrame.from_dict(updated_pop_dict)
                if pop_df.index[0] != '1':
                    # convert all indices to strings + 1
                    pop_df.index = [str(x) for x in range(1, len(pop_df) + 1)]
            updated_pop_dict = pop_df.to_dict()
            if is_cat_f_assoc:
                if is_num_f_assoc:
                    all_dfs = [pd.DataFrame(data_list[0][3], columns=[data_list[0][2]]) for data_list in data_lists]
                    all_dfs.insert(0, pop_df)
                else:
                    all_dfs = [pd.DataFrame(data_list[1], columns=[data_list[0]]) for data_list in data_lists]
                    all_dfs.insert(0, pop_df)
            else:
                all_dfs = [pop_df]

            # draw hist/box
            main_chart = make_num_dist_box(all_dfs, new_f_name, cat_assoc_radio, output_chart) if not is_num_f_assoc \
            else make_num_corr_box(all_dfs, new_f_name, num_assoc, feature_assoc, output_chart, df)

        elif data_type == 'time':
            if size is None:
                return create_list, None, None, None
            # find the input div
            num_input_children = None
            alert = no_update
            for key in ctx.callback_context.states:
                if is_dynamic_input_num_div(key):
                    num_input_children = ctx.callback_context.states[key]

            # parse deg acc and type of number
            is_dp, deg_of_acc, is_float, is_non_negative = parse_number_type(num_input_div, 3)

            # parse func info
            func_info, alert, is_valid = parse_func_info(num_input_children)
            if not is_valid:
                return alert, None, None, None

            # create data
            data_lists, alert, is_valid = make_times_data_list(func_info, size)

            # round data
            x_created_values = data_lists[0][0]
            x_smooth_values = data_lists[1][0]
            x_name = func_info[0][2]
            y_created_values = data_lists[0][1]
            y_smooth_values = data_lists[1][1]
            y_name = new_f_name
            y_created_values = round_list(is_dp, deg_of_acc, None, is_float, is_non_negative, None, y_created_values, True)
            x_created_values = round_list(is_dp, deg_of_acc, None, is_float, is_non_negative, None, x_created_values, True)

            new_data_dict = {x_name: x_created_values, y_name: y_created_values}

            new_pd = pd.DataFrame(new_data_dict)

            # create chart side by side with table
            main_chart = create_times_div(new_pd, x_name, y_name, x_smooth_values, y_smooth_values)

            # add an alert that if the data is poly0 with noise then the y-axis changes
            alert = poly_zero_alert if func_info[0][0] == 'poly0' and is_valid else no_update

            # convert all indices to strings + 1
            new_pd.index = [str(x) for x in range(1, len(new_pd) + 1)]
            updated_pop_dict = new_pd.to_dict()
    return alert, updated_pop_dict, dbc.Row(main_chart), index_display_data






