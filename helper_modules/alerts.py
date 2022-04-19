import dash_bootstrap_components as dbc
import dash_html_components as html

alert_duration = 3000

not_new_df = dbc.Alert(
    html.H5('You must enter a new or valid name and size...'),
    color='warning',
    fade=True,
    duration=alert_duration
)

dom_max_error = dbc.Alert(
    html.H5('The maxmum must be larger than the minimum...'),
    color='warning',
    fade=True,
    duration=alert_duration
)

poly_zero_alert = dbc.Alert(
    html.H5('Notice the y-axis scale has changed...'),
    color='warning',
    fade=True,
    duration=alert_duration
)

must_be_integer = dbc.Alert(
    html.H5('The size must be an integer...'),
    color='warning',
    fade=True,
    duration=alert_duration
)

some_values_none = dbc.Alert(
    html.H5("You can't have empty table cells..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
prob_error = dbc.Alert(
    html.H5("All probabilities must sum to 1..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
val_error = dbc.Alert(
    html.H5("All probabilities must be numbers.."),
    color='warning',
    fade=True,
    duration=alert_duration
)
max_min_val = dbc.Alert(
    html.H5("The minimum value must be less than the maximum..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
sd_pos = dbc.Alert(
    html.H5("The SD must be larger than zero..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
ex_pos = dbc.Alert(
    html.H5("The scale of an exponential distribution must be larger than zero..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
not_nums = dbc.Alert(
    html.H5("All percentiles must be numbers..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
val_freq_error = dbc.Alert(
    html.H5("All frequencies must be integers.."),
    color='warning',
    fade=True,
    duration=alert_duration
)
not_ascending = dbc.Alert(
    html.H5("All percentiles must be in be in ascending order..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
missing_parameters = dbc.Alert(
    html.H5("You must enter all parameters..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
freq_error_single = dbc.Alert(
    html.H5("Your total frequency must be the same as the number of spreadsheet rows..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
problem_appending_data = dbc.Alert(
    html.H5("Problem appending data..."),
    color='warning',
    fade=True,
    duration=alert_duration
)

freq_error_assoc = dbc.Alert(
    html.H5("Your total frequencies must be the same as fequencies of the asscoaicted feature...see the number in brackets..."),
    color='warning',
    fade=True,
    duration=alert_duration
)

must_name_feature = dbc.Alert(
    html.H5("You must name the feature to continue..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
duplicate_feature_name = dbc.Alert(
    html.H5("You already have a feature with this name..."),
    color='warning',
    fade=True,
    duration=alert_duration
)
unknown_error = dbc.Alert(
    html.H5("Unknown error occured with spreadsheet, create a new one to continue..."),
    color='warning',
    fade=True,
    duration=alert_duration
)

create_list = dbc.Alert(
    html.H5("Create new spreadsheet to continue..."),
    color='warning',
    fade=True,
    duration=alert_duration
)


data_successfully_appended = dbc.Alert(
    html.H5("The data has been appended to the spreadsheet..."),
    color='success',
    fade=True,
    duration=alert_duration
)