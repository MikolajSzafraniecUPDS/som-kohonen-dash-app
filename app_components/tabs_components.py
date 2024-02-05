"""
Definitions of components for each tab of the dashboard.
"""

import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
from dash import dcc, html
from app_components.callbacks import generate_som_image, ALPHA_CHANNEL_OPTIONS_ENABLED
from SOM.SOM import SelfOrganizingMap, NeighbourhoodType, LearningRateDecay


def render_som_setup_and_results_div(som: SelfOrganizingMap) -> html.Div:
    """
    Render content of main tab, containing setup of SOM
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
                            className="som-image",
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
                                    dbc.Label("Number of learning epochs", align='center', width='50%'),
                                    dcc.Slider(
                                        min=2,
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
                                            for i in [2, 25, 50, 100, 200, 300, 400, 500]
                                        }
                                    )
                                ],
                                width=9
                            ),
                            dbc.Col([
                                dbc.Label("Refresh rate:"),
                                dbc.Select(
                                    [i + 1 for i in range(20)],
                                    value=5,
                                    id="img-refresh-frequency"
                                )
                            ]),
                        ]),
                        className="learning-specification"
                    )
                ],
                    justify="center", align="center"
                ),
                html.Br(),
                html.Div([
                    dbc.Row([
                        dbc.Col([
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
                                )
                            ])
                        ], width=9),
                        dbc.Col([
                            html.Div([
                                dbc.Progress(
                                    value=0, id="learning-progress-bar",
                                    className="hidden-component"
                                )
                            ])
                        ])
                    ],
                        justify="center",
                        align="center"
                    ),
                ],
                    className="learning-specification"
                )
            ]),
            dbc.Col(
                [
                    html.Div([
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
                            max=250,
                            step=10,
                            value=100,
                            id="som-size-slider",
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True
                            },
                            marks={
                                i: "{}".format(i)
                                for i in [10, 30, 50, 100, 150, 200, 250]
                            }
                        ),
                        html.Br(),
                        dbc.Label("Include alpha channel"),
                        dbc.RadioItems(
                            options=ALPHA_CHANNEL_OPTIONS_ENABLED,
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
                                "always_visible": True,
                                "template": "{value}%"
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
                            dbc.Checkbox(
                                id="show-rgba-range-sliders",
                                label="Show RGBA range sliders",
                                value=False
                            ),
                            dbc.Collapse(
                                dbc.Card([
                                    dbc.Label("Red component range"),
                                    dcc.RangeSlider(
                                        id="red-range-slider",
                                        min=0,
                                        max=255,
                                        step=1,
                                        value=[0, 255],
                                        marks={i: "{}".format(i) for i in [1, 10, 50, 100, 150, 200, 255]},
                                        allowCross=False,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    html.Br(),
                                    dbc.Label("Green component range"),
                                    dcc.RangeSlider(
                                        id="green-range-slider",
                                        min=0,
                                        max=255,
                                        step=1,
                                        value=[0, 255],
                                        marks={i: "{}".format(i) for i in [1, 10, 50, 100, 150, 200, 255]},
                                        allowCross=False,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    html.Br(),
                                    dbc.Label("Blue component range"),
                                    dcc.RangeSlider(
                                        id="blue-range-slider",
                                        min=0,
                                        max=255,
                                        step=1,
                                        value=[0, 255],
                                        marks={i: "{}".format(i) for i in [1, 10, 50, 100, 150, 200, 255]},
                                        allowCross=False,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    html.Br(),
                                    dbc.Label("Alpha channel component range"),
                                    dcc.RangeSlider(
                                        id="alpha-channel-range-slider",
                                        min=0,
                                        max=255,
                                        step=1,
                                        value=[0, 255],
                                        marks={i: "{}".format(i) for i in [1, 10, 50, 100, 150, 200, 255]},
                                        allowCross=False,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ]),
                                id="rgba-range-sliders-collapse",
                                is_open=False
                            )
                        ]),
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
                ]
            )
        ])
    ])

    return res


def render_about_learning_params_tab() -> html.Div:
    """
    Render content of Tab containing information about
    learning parameters

    :return: output tab Div
    """
    res = html.Div([
        html.Br(),
        dbc.Col([
            dbc.Row([
                html.Br(),
                html.H2("About SOM"),
                html.P([
                    """
                    Self organizing map (Kohonen network) is an unsupervised machine learning 
                    algorithm and a type of self-organizing neural network, allowing to produce a 
                    low-dimensional representation of a higher dimensional data set.
                    Self organizing maps can help us understand the relationships between objects described by many 
                    variables whose abstract nature is difficult to grasp with intuition.  Kohonen networks have a 
                    specific topological structure (e.g. the form of a two-dimensional grid composed of individual neurons). 
                    The neurons are initialised with random weights, but in the course of learning the weights are modified 
                    so that neurons lying close to each other represent similar objects from the multidimensional space 
                    from which our learning examples are drawn. However, producing a correct 
                    mapping requires the right choice of learning parameters, depending on our requirements, network 
                    size, data dimensionality, etc. This application has been developed to help understand the relationships 
                    between the parameters and to support in the process of selecting their values. Using the RGB colour 
                    palette, it helps to visually check that the set of parameters we have chosen does not produce
                    network sensitive only to one type of signal (too uniform, single-colour image) or or a network 
                    without seamless transitions between classes (too sharp borders between colors). 
                    According to my experience, this way of interacting with Kohonen networks can also be great fun : ) 
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            dbc.Row([
                html.Br(),
                html.Hr(),
                html.H4("Idea behind RGB(A) self-organizing map"),
                html.P([
                    """
                    Our RGB(A) Kohonen network is a square-shaped, two dimensional network in topological
                    terms. The length of the side of the square is determined in the settings and can be 
                    between 10 and 250 neurons (pixels). Each neuron has three or four weights, depending 
                    on whether we want to use the alpha channel component. Initial weights are assigned randomly
                    as integers from range (0, 255). The network can be visualised as a PNG image - each neuron 
                    represents an individual pixel and its weights are the value of the RGB(A) components.
                    The aim is to train the network to produce the widest possible colour palette while 
                    maintaining smooth transitions between them. The result of a network with too uniform a 
                    colour or too sharp colour transitions indicates that we should probably change the learning parameters.
                    The initial state of the network and examples of correct and incorrect learning results are presented below.
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label(
                        "Initial state of network",
                        className="tutorial-image-label"
                    ),
                    html.Img(
                        src="assets/initial_state.png",
                        className="tutorial-image"
                    )
                ],
                    className="text-align-center",
                    width=4
                ),
                dbc.Col([
                    dbc.Label(
                        "Properly learned network",
                        className="tutorial-image-label"
                    ),
                    html.Img(
                        src="assets/properly_fitted_network.png",
                        className="tutorial-image"
                    )
                ],
                    className="text-align-center",
                    width=4
                )
            ],
                justify="evenly"
            ),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Label(
                        "Network too uniform",
                        className="tutorial-image-label"
                    ),
                    html.Img(
                        src="assets/uniform_network.png",
                        className="tutorial-image"
                    )
                ],
                    className="text-align-center",
                    width=4
                ),
                dbc.Col([
                    dbc.Label(
                        "Colors transitions too sharp",
                        className="tutorial-image-label"
                    ),
                    html.Img(
                        src="assets/borders_too_sharp.png",
                        className="tutorial-image"
                    )
                ],
                    className="text-align-center",
                    width=4
                )
            ],
                justify="evenly"
            ),
            html.Br(),
            dbc.Row([
                html.P([
                    """
                    Process of learning network is pretty simple - in each iteration we randomly draw
                    a vector containing RGB(A) components (learning example / input vector). As a next
                    step we need to find so-called BMU (Best Matching Unit) - a neuron which is the most
                    similar to the input vector in terms of euclidean distance. Then we need to calculate
                    a value of learning rate for given iteration and value of neighbourhood for each neuron
                    and BMU, using the formula we selected (gaussian, bubble or mexican hat function). Finally,
                    we update weights of each neuron. Exact formulas and details of learning process are
                    described pretty well in 
                    """,
                    html.A("this paper.", href="https://ijmo.org/vol6/504-M08.pdf", target="_empty")
                ],
                    className="justified-paragraph"
                )
            ])
        ],
            width={
                "size": 8, "offset": 1
            }
        )
    ])

    return res
