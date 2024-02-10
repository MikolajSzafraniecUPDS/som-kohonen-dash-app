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
                                width=12,
                                lg=9
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
                            dbc.Button(
                                "Run learning",
                                id="run-learning-btn",
                                color="success",
                                className="me-1",
                                disabled=False
                            ),
                        ],
                            className="d-grid gap-2",
                            lg=2,
                            width=4
                        ),
                        dbc.Col([
                            dbc.Button(
                                "Reset network",
                                id="reset-som-btn",
                                color="warning",
                                className="me-1"
                            )
                        ],
                            className="d-grid gap-2",
                            lg=2,
                            width=4
                        ),
                        dbc.Col([
                            dbc.Button(
                                "Stop learning",
                                id="stop-learning-btn",
                                color="danger",
                                className="me-1",
                                disabled=True
                            )
                        ],
                            className="d-grid gap-2",
                            lg=2,
                            width=4
                        ),
                        dbc.Col([
                            html.Div([
                                dbc.Label("Learning progress", className="text-align-center"),
                                dbc.Progress(
                                    value=0, id="learning-progress-bar"
                                )
                            ],
                                id="learning-progress-div",
                                className="hidden-component"
                            )
                        ],
                            lg=3,
                            width=8
                        )
                    ],
                        justify="center",
                        align="center"
                    ),
                ],
                    className="text-align-center"
                )
            ],
                lg=6,
                width=12
            ),
            dbc.Col(
                [
                    html.Div([
                        html.H2("SOM parameters", className="text-align-center"),
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
                            min=0.1,
                            max=100,
                            step=0.1,
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
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Neighbourhood type")
                            ],
                                lg=4,
                                width=6
                            ),
                            dbc.Col([
                                dbc.Label("Decay function for learning rate")
                            ],
                                lg=4,
                                width=6
                            )
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Select(
                                    [option.value for option in NeighbourhoodType],
                                    value="Gaussian",
                                    id="neighbourhood-type"
                                )
                            ],
                                lg=4,
                                width=6
                            ),
                            dbc.Col([
                                dbc.Select(
                                    [option.value for option in LearningRateDecay],
                                    value="Inverse of time",
                                    id="learning-rate-decay-func"
                                )
                            ],
                                lg=4,
                                width=6
                            )
                        ]),
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
                ],
                width=12,
                lg=6
            )
        ]),
        html.Br()
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
                        className="tutorial-image-network"
                    )
                ],
                    className="text-align-center",
                    lg=4,
                    width=6
                ),
                dbc.Col([
                    dbc.Label(
                        "Properly learned network",
                        className="tutorial-image-label"
                    ),
                    html.Img(
                        src="assets/properly_fitted_network.png",
                        className="tutorial-image-network"
                    )
                ],
                    className="text-align-center",
                    lg=4,
                    width=6
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
                        className="tutorial-image-network"
                    )
                ],
                    className="text-align-center",
                    lg=4,
                    width=6
                ),
                dbc.Col([
                    dbc.Label(
                        "Colors transitions too sharp",
                        className="tutorial-image-label"
                    ),
                    html.Img(
                        src="assets/borders_too_sharp.png",
                        className="tutorial-image-network"
                    )
                ],
                    className="text-align-center",
                    lg=4,
                    width=6
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
            ]),
            dbc.Row([
                html.Br(),
                html.Hr(),
                html.H4("Learning procedure"),
                html.P([
                    """
                    In order to start learning procedure all you need to do is just click 'Run learning'
                    button - you can do it directly after launching an app.
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="assets/learning_details_1.png",
                        className="tutorial-image-screen"
                    )
                ],
                    lg=7,
                    width=12
                )
            ],
                justify="center"
            ),
            html.Br(),
            dbc.Row([
                html.P([
                    """
                    If you would like to change a parameters you need to apply changes using
                    'Update network' button.
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="assets/learning_details_2.png",
                        className="tutorial-image-screen"
                    )
                ],
                    lg=7,
                    width=12
                )
            ],
                justify="center"
            ),
            html.Br(),
            dbc.Row([
                html.P([
                    """
                    Please note that if you change size of the network or alpha channel indicator
                    network will be reset as well as your learning results. If you change some parameters
                    but you want to discard changes you can use 'Reset settings changes' button.
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="assets/learning_details_3.png",
                        className="tutorial-image-screen"
                    )
                ],
                    lg=7,
                    width=12
                )
            ],
                justify="center"
            ),
            html.Br(),
            dbc.Row([
                html.P([
                    """
                    If you have already start learning and you wish to stop the process you can
                    use 'Stop learning' button.
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="assets/learning_details_4.png",
                        className="tutorial-image-screen"
                    )
                ],
                    lg=8,
                    width=12
                )
            ],
                justify="center"
            ),
            html.Br(),
            dbc.Row([
                html.P([
                    """
                    You can follow the progress of your learning with the progress bar.
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="assets/learning_details_5.png",
                        className="tutorial-image-screen"
                    )
                ],
                    lg=8,
                    width=12
                )
            ],
                justify="center"
            ),
            html.Br(),
            dbc.Row([
                html.P([
                    """
                    If you want to reset network after learning procedure you can use 'Reset network'
                    button.
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="assets/learning_details_6.png",
                        className="tutorial-image-screen"
                    )
                ],
                    lg=8,
                    width=12
                )
            ],
                justify="center"
            ),
            html.Br(),
            dbc.Row([
                html.Hr(),
                html.H4("Learning parameters"),
                html.P([
                    """
                    All learning parameters are described below.
                    """
                ],
                    className="justified-paragraph"
                )
            ]),
            html.Br(),
            dbc.Row([
                dbc.Accordion([
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                This parameter simply defines the size of the network - the number
                                of neurons (pixels) per side of the grid.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_1.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=9,
                                    width=12
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="SOM size"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                With this parameter, you can decide whether the network should contain 
                                an alpha channel, which determines the transparency of a pixel.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_2.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=3,
                                    width=6
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="Include alpha channel"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                This parameter defines the initial neighbourhood radius as a percentage 
                                of the network size. Too small a value for this parameter may result in 
                                only small parts of our map being taught. Too large a value will result 
                                in an almost uniform map after each learning epoch.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_3.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=9,
                                    width=12
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="Initial neighbourhood radius"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                Learning rate is one of the key parameters for learning. It 
                                defines the strength with which the neuron weights are modified 
                                in a given learning epoch. This parameter indicates the initial 
                                value of the learning rate - this is then reduced with each 
                                subsequent epoch.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_4.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=9,
                                    width=12
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="Initial learning rate"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                This function is closely related to the neighbourhood radius - using 
                                its value for a given learning epoch, it calculates the neighbourhood 
                                value for each neuron relative to the BMU. By using the Gaussian 
                                neighbourhood function, we guarantee that the weights of each neuron 
                                will be modified, to a greater extent the closer a given neuron is 
                                to the BMU in terms of network topology. The Bubble neighbourhood 
                                function equally modifies the weights of only those neurons that 
                                are located within a given radius relative to the BMU.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_5.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=4,
                                    width=8
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="Neighbourhood type"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                As mentioned above, the value of the learning rate decreases 
                                with each learning epoch - so that specific regions of the 
                                network can 'specialise' in detecting a signal of a particular type. 
                                This function defines a learning rate decrease depending on 
                                the learning epoch number.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_6.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=4,
                                    width=8
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="Decay function for learning rate"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                These sliders allow you to define the RGB(A) channel 
                                ranges that will be passed to the network as learning 
                                examples. For example, leaving the range high for the 
                                red channel and lowering it for the green and blue 
                                channels will make the learning results strongly red-shifted.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_7.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=8,
                                    width=12
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="RGB(A) range sliders"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                This slider allows you to specify the number of learning epochs.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_8.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=8,
                                    width=12
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="Number of learning epochs"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            html.P([
                                """
                                Refresh rate of image during network learning. Too high a frequency (1-5) 
                                for large networks can significantly prolong the learning process.
                                """
                            ],
                                className="justified-paragraph"
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Img(
                                        src="assets/som_params_9.png",
                                        className="tutorial-image-screen"
                                    )
                                ],
                                    lg=4,
                                    width=8
                                )
                            ],
                                justify="center"
                            )
                        ]),
                        title="Refresh rate"
                    )
                ],
                    flush=True,
                    active_item=None,
                    always_open=True
                )
            ]),
            dbc.Row([
                html.Br(),
                html.Hr(),
                html.H4("References"),
                html.P([
                    """
                    If you would like to better understand the process of learning SOM and
                    take a look at exact formulas for parameters described above I highly
                    recommend to take a look at materials below:
                    """
                ],
                    className="justified-paragraph"
                ),
                html.A(
                    """
                    1) Wiki article
                    """,
                    href="https://en.wikipedia.org/wiki/Self-organizing_map",
                    target="_empty"
                ),
                html.A(
                    """
                    2) Appropriate Learning Rate and Neighborhood Function of
                    Self-organizing Map (SOM) for Specific Humidity Pattern
                    Classification over Southern Thailand
                    """,
                    href="https://ijmo.org/vol6/504-M08.pdf",
                    target="_empty"
                ),
                html.A(
                    """
                    3) Self Organizing Maps - TowardsDataScience article
                    """,
                    href="https://towardsdatascience.com/self-organizing-maps-1b7d2a84e065",
                    target="_empty"
                ),
                html.A(
                    """
                    4) Nice presentation containing formula for Mexican Hat neighbourhood function
                    """,
                    href="https://coursepages2.tuni.fi/tiets07/wp-content/uploads/sites/110/2019/01/Neurocomputing3.pdf",
                    target="_empty"
                )
            ]),
            html.Br()
        ],
            lg={
                "size": 8, "offset": 1
            },
            width={
                "size": 12, "offset": 1
            }
        )
    ])

    return res
