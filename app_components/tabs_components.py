"""
Definitions of components for each tab of the dashboard.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html
from app_components.callbacks import generate_som_image
from SOM.SOM import SelfOrganizingMap


def render_som_setup_and_results_div(som: SelfOrganizingMap) -> html.Div:
    """
    Render content of main tab, containing setup of our SOM
    and presenting results in the form of RGB(A) image.

    :param som: SelfOrganizingMap object to print
    :return: output tab Div
    """
    res = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    html.Img(
                        id="som-img",
                        style={
                            "display": "block",
                            "margin-left": "auto",
                            "margin-right": "auto",
                            "width": "90%",
                            "image-rendering": "pixelated"
                        },
                        src=generate_som_image(som)
                    ),
                    html.Br(),
                    html.Div(
                        [
                            dbc.Label("Number of learning iterations", align='center', width='50%'),
                            dcc.Slider(
                                min=1,
                                max=500,
                                step=1,
                                value=50,
                                id="number-of-iterations-learning",
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": True
                                },
                                marks={
                                    i: "{}".format(i)
                                    for i in [1, 25, 50, 100, 200, 300, 400, 500]
                                }
                            )
                        ],
                        style={
                            'width': '80%',
                            'text-align': 'center'
                        }
                    )
                ],
                    justify="center", align="center"
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "Run learning",
                            id="run-learning-btn",
                            color="success",
                            className="me-1",
                            disabled=False
                        )
                    ]),
                    dbc.Col([
                        dbc.Button(
                            "Reset network",
                            id="reset-som-btn",
                            color="danger",
                            className="me-1"
                        )
                    ])
                ])
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
                        {"label": "True", "value": True},
                        {"label": "False", "value": False},
                    ],
                    value=False,
                    id="include-alpha-channel"
                ),
                html.Br(),
                dbc.Button(
                    "Update network",
                    id="update-network-btn",
                    color="primary",
                    className="me-1",
                    disabled=True
                )
            ])
        ])
    ])

    return res
