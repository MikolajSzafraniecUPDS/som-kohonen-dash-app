"""
Definitions of components for each tab of the dashboard.
"""

import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
from dash import dcc, html
from app_components.callbacks import generate_som_image
from SOM.SOM import SelfOrganizingMap, NeighbourhoodType, LearningRateDecay


def render_som_setup_and_results_div(som: SelfOrganizingMap) -> html.Div:
    """
    Render content of main tab, containing setup of our SOM
    and presenting results in the form of RGB(A) image.

    :param som: SelfOrganizingMap object to print
    :return: output tab Div
    """
    res = html.Div([
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dls.Hash(
                        html.Img(
                            id="som-img",
                            style={
                                "display": "block",
                                "margin-left": "auto",
                                "margin-right": "auto",
                                "width": "80%",
                                "image-rendering": "pixelated"
                            },
                            src=generate_som_image(som)
                        ),
                        color="#435278",
                        speed_multiplier=2,
                        size=100
                    ),
                    html.Br(),
                    html.Div(
                        dbc.Row([
                            dbc.Col(
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
                                width=9
                            ),
                            dbc.Col([
                                dbc.Label("Refresh rate:"),
                                dbc.Select(
                                    [i+1 for i in range(20)],
                                    value=5,
                                    id="img-refresh-frequency"
                                )
                            ]),
                        ]),
                        style={
                            'width': '80%',
                            'text-align': 'center'
                        }
                    )
                ],
                    justify="center", align="center"
                ),
                html.Br(),
                dbc.Row([
                    html.Div([
                        dbc.Button(
                            "Run learning",
                            id="run-learning-btn",
                            color="success",
                            className="me-1",
                            disabled=False
                        ),
                        dbc.Button(
                            "Reset network",
                            id="reset-som-btn",
                            color="warning",
                            className="me-1"
                        ),
                        dbc.Button(
                            "Stop learning",
                            id="stop-learning-btn",
                            color="danger",
                            className="me-1",
                            disabled=True
                        ),
                        html.Div([
                            dbc.Progress(value=0, id="learning-progress-bar")
                        ],
                            style={"visibility": "hidden"},
                            id="learning-progress-div"
                        )
                    ],
                        style={
                            'width': '75%',
                        }
                    )
                ],
                    justify="center",
                    align="center"
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
                        {"label": "True", "value": True},
                        {"label": "False", "value": False},
                    ],
                    value=False,
                    id="include-alpha-channel"
                ),
                html.Br(),
                dbc.Label("Initial neighbourhood radius"),
                dcc.Slider(
                    min=1,
                    max=100,
                    step=1,
                    value=10,
                    id="initial-neighbourhood-radius",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True
                    },
                    marks={
                        i: {'label': "{0}%".format(i)}
                        for i in [1, 5, 10, 25, 50, 75, 100]
                    }
                ),
                html.Br(),
                dbc.Label("Initial learning rate"),
                dcc.Slider(
                    min=0.01,
                    max=1.00,
                    step=0.01,
                    value=0.5,
                    id="initial-learning-rate",
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True
                    },
                    marks={
                        i: {'label': "{0}".format(i)}
                        for i in [
                            0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
                        ]
                    }
                ),
                html.Br(),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Neighbourhood type"),
                            dbc.Select(
                                [option.value for option in NeighbourhoodType],
                                value="Gaussian",
                                id="neighbourhood-type"
                            )
                        ]),
                        dbc.Col([
                            dbc.Label("Decay function for learning rate"),
                            dbc.Select(
                                [option.value for option in LearningRateDecay],
                                value="Inverse of time",
                                id="learning-rate-decay-func"
                            )
                        ])
                    ])
                ], style={"width": "60%"}),
                html.Br(),
                html.Div([
                    dbc.Button(
                        "Update network",
                        id="update-network-btn",
                        color="primary",
                        className="me-1",
                        disabled=True
                    ),
                    dbc.Button(
                        "Reset settings changes",
                        id="reset-settings-changes-btn",
                        color="danger",
                        className="me-1",
                        disabled=True
                    )
                ])
            ])
        ])
    ])

    return res
