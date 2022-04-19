import numpy as np
import dash_html_components as html

THEME_BREAK = html.Hr(style={'border-top': '1px dotted red'})

COLOUR_SCHEME = ['Light', 'Dark', 'Pastel', 'Antique', 'Bold', 'Safe', 'Vivid']

ALL_COLOURS = [
    ['very light red','lightsalmon'],
    ['light red','lightcoral'],
    ['red','red'],
    ['dark red','darkred'],
    ['black','black'],
    ['gray','gray'],
    ['dark gray','darkgray'],
    ['light grey','lightgrey'],
    ['dark green','darkgreen'],
    ['green','green'],
    ['light green','lightgreen'],
    ['turquoise','turquoise'],
    ['teal','teal'],
    ['aqua','aqua'],
    ['cyan','cyan'],
    ['light purple','thistle'],
    ['violet','violet'],
    ['purple','purple'],
    ['magenta','magenta'],
]

MARKER_STYLES = np.array(
    ['circle', 'diamond', 'triangle-up', 'pentagon', 'star', 'bowtie', 'asterisk', 'hash', 'octagon'])

LINE_STYLES = np.array(
    ['solid','dash','dot','dashdot'])

ROW_MARGIN = '10px'
ROW_HEIGHT_NUM = '100px'
COLUMN_PAD = '50px'

PAGE_BOTTOM_PAD = '150px'

RED = 'rgba(238, 157, 148, 0.7)'
BLUE = 'rgba(157, 187, 227, 0.7)'

FUNCTIONS = ['poly0', 'poly1', 'poly2', 'poly3', 'exp', 'ln', 'sin', 'cos', 'tan', 'logistic']
PARAMS = [1, 2, 3, 4, 4, 4, 4, 4, 4, 4]
