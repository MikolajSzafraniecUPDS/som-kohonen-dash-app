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
        dbc.Row([
            dbc.Col([
                html.Img(
                    id="som-img",
                    style={
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "width": "90%",
                        "image-rendering": "pixelated"
                    }
                )
            ]),
            dbc.Col([
                html.H2("SOM parameters"),
                html.P(
                    """
                    Set of parameters defining Kohonen network (size, whether to include
                    alpha channel or not) and learning process (learning rate, neighbourhood
                    function, etc.). 
                    """,
                    className="lead"
                ),
                html.Br(),
                dbc.Label("SOM size"),
                dcc.Slider(
                    min=10,
                    max=500,
                    step=10,
                    value=100,
                    id="som-size-slider",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True
                    },
                    marks={
                        i: "{}".format(i)
                        for i in [10, 50, 100, 250, 500]
                    }
                ),
                html.Br(),
                dbc.Label("Include alpha channel"),
                dbc.RadioItems(
                    options=[
                        {"label": "True", "value": 1},
                        {"label": "False", "value": 2},
                    ],
                    value=2,
                    id="include-alpha-channel"
                )
            ])
        ])
    ])

    return res
