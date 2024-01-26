"""
Definitions of components for each tab of the dashboard.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

def render_som_setup_and_results_div() -> html.Div:
    """
    Render content of main tab, containing setup of our SOM
    and presenting results in the form of RGB(A) image.

    :return: output tab Div
    """
    res = html.Div([
        html.Img(
            id="som-img",
            style={
                "display": "block",
                "margin-left": "auto",
                "margin-right": "auto",
                "width": "90%"
            }
        )
    ])