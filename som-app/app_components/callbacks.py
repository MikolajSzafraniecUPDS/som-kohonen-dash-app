"""
Definitions of callbacks functions fot the Dash applications. In order to
make the main app.py file more concise we are going to define callbacks
in this file.
"""

import base64
import time

import dash
from dash.dash import Dash
from dash import Input, Output, State
from SOM.SOM import SelfOrganizingMap, NeighbourhoodType, LearningRateDecay
from io import BytesIO
from app_components.utils import get_som_from_cache, store_som_in_cache, rm_som_from_cache
from typing import List

ALPHA_CHANNEL_OPTIONS_ENABLED = [
    {"label": "True", "value": True},
    {"label": "False", "value": False},
]

ALPHA_CHANNEL_OPTIONS_DISABLED = [
    {"label": "True", "value": True, "disabled": True},
    {"label": "False", "value": False, "disabled": True},
]


def generate_som_image(som: SelfOrganizingMap) -> str:
    """
    Generate encoded SOM image

    :param som: SelfOrganizingMap object to print
    :return: encoded PNG image as string
    """
    som_im = som.to_image()
    buff = BytesIO()
    som_im.save(buff, format="PNG")
    im_encoded = 'data:image/png;base64,{}'.format(base64.b64encode(buff.getvalue()).decode("utf-8"))

    return im_encoded


def get_callbacks(app: Dash) -> None:
    """
    Definitions of callback functions for the Dash application

    :param app: Dash app object
    :param som: SelfOrganizingMap object, representing Kohonen Network
    """

    @app.callback(
        Output("update-network-btn", "disabled", allow_duplicate=True),
        inputs=[
            Input("som-size-slider", "value"),
            Input("include-alpha-channel", "value"),
            Input("initial-neighbourhood-radius", "value"),
            Input("initial-learning-rate", "value"),
            Input("neighbourhood-type", "value"),
            Input("learning-rate-decay-func", "value"),
            Input("red-range-slider", "value"),
            Input("green-range-slider", "value"),
            Input("blue-range-slider", "value"),
            Input("alpha-channel-range-slider", "value")
        ],
        state=State("session-id", "children"),
        prevent_initial_call=True
    )
    def update_button_status(
            som_size: int,
            include_alpha_channel: bool,
            initial_neighbourhood_radius: int,
            initial_learning_rate: float,
            neighbourhood_type: str,
            learning_rate_decay_func: str,
            red_range: List[int],
            green_range: List[int],
            blue_range: List[int],
            alpha_channel_range: List[int],
            session_id: str,
    ):
        """
        Check whether are there any changes in som parameters and
        enable update button if there are

        :param som_size: network size
        :param include_alpha_channel: alpha channel indicator
        :param initial_neighbourhood_radius: initial neighbourhood radius as a percent
            of network radius size
        :param initial_learning_rate: initial learning rate
        :param neighbourhood_type: type of neighbourhood function to use - one
            of 'Gaussian' or 'Bubble'
        :param learning_rate_decay_func: type of decay function for learning rate. Possible
            choices are 'Linear', 'Inverse of time' and 'Power series'.
        :param red_range: range of red component
        :param green_range: range of green component
        :param blue_range: range of blue component
        :param alpha_channel_range: alpha channel of red component
        :param session_id: id of current session
        :param session_id: id of current session
        """
        som = get_som_from_cache(session_id)
        size_same = som_size == som.size
        alpha_channel_same = include_alpha_channel == som.include_alpha_channel
        neighbourhood_radius_same = (initial_neighbourhood_radius/100) == som.initial_neighbourhood_radius
        learning_rate_same = initial_learning_rate == som.initial_learning_rate
        neighbourhood_type_same = neighbourhood_type == som.neighbourhood_type.value
        learning_rate_decay_func_same = learning_rate_decay_func == som.learning_rate_decay_func.value

        rgba_low = (
            red_range[0],
            green_range[0],
            blue_range[0],
            alpha_channel_range[0]
        )

        rgba_high = (
            red_range[1],
            green_range[1],
            blue_range[1],
            alpha_channel_range[1]
        )

        rgba_low_same = rgba_low == som.rgba_low
        rgba_high_same = rgba_high == som.rgba_high

        update_button_disabled = all(
            [size_same, alpha_channel_same, neighbourhood_radius_same,
             learning_rate_same, neighbourhood_type_same, learning_rate_decay_func_same,
             rgba_low_same, rgba_high_same]
        )

        return update_button_disabled

    @app.callback(
        [
            Output("som-img", "src", allow_duplicate=True),
            Output("update-network-btn", "disabled", allow_duplicate=True)
        ],
        inputs=Input("update-network-btn", "n_clicks"),
        state=[
            State("som-size-slider", "value"),
            State("include-alpha-channel", "value"),
            State("initial-neighbourhood-radius", "value"),
            State("initial-learning-rate", "value"),
            State("neighbourhood-type", "value"),
            State("learning-rate-decay-func", "value"),
            State("red-range-slider", "value"),
            State("green-range-slider", "value"),
            State("blue-range-slider", "value"),
            State("alpha-channel-range-slider", "value"),
            State("session-id", "children"),
        ],
        # background=True,
        # running=[
        #     (Output("update-network-btn", "disabled", allow_duplicate=True), True, True)
        # ],
        prevent_initial_call=True
    )
    def update_network(
            n_clicks: int,
            som_size: int,
            include_alpha_channel: bool,
            initial_neighbourhood_radius: int,
            initial_learning_rate: float,
            neighbourhood_type: str,
            learning_rate_decay_func: str,
            red_range: List[int],
            green_range: List[int],
            blue_range: List[int],
            alpha_channel_range: List[int],
            session_id: str,
    ):
        """
        Update network and print its image representation in app

        :param n_clicks: number of button clicks
        :param som_size: size of network
        :param include_alpha_channel: value indicating whether alpha channel should be
            included
        :param initial_neighbourhood_radius: initial neighbourhood radius as a percent
            of network radius size
        :param initial_learning_rate: initial learning rate
        :param neighbourhood_type: type of neighbourhood function to use - one
            of 'Gaussian' or 'Bubble'
        :param learning_rate_decay_func: type of decay function for learning rate. Possible
            choices are 'Linear', 'Inverse of time' and 'Power series'.
        :param red_range: range of red component
        :param green_range: range of green component
        :param blue_range: range of blue component
        :param alpha_channel_range: alpha channel of red component
        :param session_id: id of current session
        """
        som = get_som_from_cache(session_id)
        current_size = som.size
        current_alpha_channel_indicator = som.include_alpha_channel

        size_different = current_size != som_size
        alpha_channel_different = current_alpha_channel_indicator != include_alpha_channel

        # If both size and alpha channel indicator are going to be updated we
        # don't want to reset network twice
        if size_different and alpha_channel_different:
            som.resize_and_update_alpha_channel_indicator(
                som_size, include_alpha_channel
            )
        else:
            som.size = som_size
            som.include_alpha_channel = include_alpha_channel

        som.initial_neighbourhood_radius = initial_neighbourhood_radius/100
        som.initial_learning_rate = initial_learning_rate
        som.neighbourhood_type = NeighbourhoodType(neighbourhood_type)
        som.learning_rate_decay_func = LearningRateDecay(learning_rate_decay_func)

        rgba_low = (
            red_range[0],
            green_range[0],
            blue_range[0],
            alpha_channel_range[0]
        )

        rgba_high = (
            red_range[1],
            green_range[1],
            blue_range[1],
            alpha_channel_range[1]
        )

        som.rgba_low = rgba_low
        som.rgba_high = rgba_high

        store_som_in_cache(session_id, som)

        if size_different or alpha_channel_different:
            som_img = generate_som_image(som)
        else:
            som_img = dash.no_update

        return som_img, True

    @app.callback(
        [
            Output("reset-settings-changes-btn", "disabled"),
            Output("run-learning-btn", "disabled"),
            Output("reset-som-btn", "disabled")
        ],
        Input("update-network-btn", "disabled")
    )
    def buttons_disabled_enabled(update_network_btn_disabled: bool):
        """
        Change status of buttons 'disabled' property

        :param update_network_btn_disabled: is 'update-network-btn' disabled
        """
        reset_settings_disabled = update_network_btn_disabled
        learn_reset_network_btns_disabled = not update_network_btn_disabled
        return reset_settings_disabled, learn_reset_network_btns_disabled, learn_reset_network_btns_disabled

    @app.callback(
        [
            Output("som-size-slider", "value"),
            Output("include-alpha-channel", "value"),
            Output("initial-neighbourhood-radius", "value"),
            Output("initial-learning-rate", "value"),
            Output("neighbourhood-type", "value"),
            Output("learning-rate-decay-func", "value"),
            Output("red-range-slider", "value", allow_duplicate=True),
            Output("green-range-slider", "value", allow_duplicate=True),
            Output("blue-range-slider", "value", allow_duplicate=True),
            Output("alpha-channel-range-slider", "value", allow_duplicate=True),
            Output("update-network-btn", "disabled", allow_duplicate=True),
        ],
        inputs=Input("reset-settings-changes-btn", "n_clicks"),
        state=State("session-id", "children"),
        prevent_initial_call=True
    )
    def reset_settings_changes(n_clicks: int, session_id: str):
        """
        Reset changes of network settings

        :param n_clicks: how many times reset button was clicked
        :param session_id: id of current session
        """
        som = get_som_from_cache(session_id)
        som_size = som.size
        som_alpha_channel_indicator = som.include_alpha_channel
        som_initial_neighbourhood_radius = som.initial_neighbourhood_radius*100
        som_initial_learning_rate = som.initial_learning_rate
        som_neighbourhood_type = som.neighbourhood_type.value
        som_learning_rate_decay_func = som.learning_rate_decay_func.value
        rgba_low = som.rgba_low
        rgba_high = som.rgba_high

        red_low_high = [rgba_low[0], rgba_high[0]]
        green_low_high = [rgba_low[1], rgba_high[1]]
        blue_low_high = [rgba_low[2], rgba_high[2]]
        alpha_channel_low_high = [rgba_low[2], rgba_high[2]]

        return (
            som_size,
            som_alpha_channel_indicator,
            som_initial_neighbourhood_radius,
            som_initial_learning_rate,
            som_neighbourhood_type,
            som_learning_rate_decay_func,
            red_low_high,
            green_low_high,
            blue_low_high,
            alpha_channel_low_high,
            True
        )

    @app.callback(
        Output("som-img", "src", allow_duplicate=True),
        inputs=Input("reset-som-btn", "n_clicks"),
        state=State("session-id", "children"),
        prevent_initial_call=True
    )
    def reset_network(n_clicks: int, session_id: str):
        """
        Reset network after learning procedure

        :param n_clicks: how many times reset button was clicked
        :param session_id: id of current session
        """
        som = get_som_from_cache(session_id)
        som.reset_network()
        store_som_in_cache(session_id, som)
        som_img = generate_som_image(som)
        return som_img

    @app.callback(
        Output("app-closed", "children"),
        inputs=Input("clear_cache_btn", "n_clicks"),
        state=State("session-id", "children"),
        prevent_initial_call=True
    )
    def clear_cache(n_clicks: int, session_id: str):
        """
        Remove som object from cache when session is ended

        :param n_clicks: how many times reset button was clicked
        :param session_id: id of current session
        """
        rm_som_from_cache(session_id)
        return "File removed"

    @app.callback(
        Output("learning-progress-bar", "value", allow_duplicate=True),
        inputs=Input("run-learning-btn", "n_clicks"),
        background=True,
        state=[
            State("session-id", "children"),
            State("number-of-iterations-learning", "value"),
            State("img-refresh-frequency", "value")
        ],
        running=[
            (Output("run-learning-btn", "disabled"), True, False),
            (Output("reset-som-btn", "disabled"), True, False),
            (Output("stop-learning-btn", "disabled"), False, True),
            (
                Output("learning-progress-div","className"),
                "visible-component",
                "hidden-component"
            ),
            (Output("som-size-slider", "disabled"), True, False),
            (
                    Output("include-alpha-channel", "options"),
                    ALPHA_CHANNEL_OPTIONS_DISABLED,
                    ALPHA_CHANNEL_OPTIONS_ENABLED
            ),
            (Output("initial-neighbourhood-radius", "disabled"), True, False),
            (Output("initial-learning-rate", "disabled"), True, False),
            (Output("neighbourhood-type", "disabled"), True, False),
            (Output("learning-rate-decay-func", "disabled"), True, False),
            (Output("red-range-slider", "disabled"), True, False),
            (Output("green-range-slider", "disabled"), True, False),
            (Output("blue-range-slider", "disabled"), True, False),
            (Output("alpha-channel-range-slider", "disabled"), True, False)
        ],
        cancel=Input("stop-learning-btn", "n_clicks"),
        progress=[
            Output("learning-progress-bar", "value"),
            Output("learning-progress-bar", "label"),
            Output("som-img", "src", allow_duplicate=True)
        ],
        prevent_initial_call=True
    )
    def learn_network(
            set_progress,
            n_clicks: int,
            session_id: str,
            number_of_iterations: int,
            img_refresh_rate: int
    ):
        """
        Train network using provided settings. During learning image
        will be refreshed according to specified refresh rate.

        :param set_progress: callable - progress output
        :param n_clicks: how many times button was clicked
        :param session_id: id of user's session
        :param number_of_iterations: number of learning iterations
        :param img_refresh_rate: refresh rate of image
        """
        som = get_som_from_cache(session_id)
        number_of_iterations = int(number_of_iterations)
        img_refresh_rate = int(img_refresh_rate)
        som.number_of_iterations = number_of_iterations
        som.current_iteration = 1
        for i in range(number_of_iterations):
            som.train_network_single_iteration()
            progress_perc = int(((i+1)/number_of_iterations)*100)
            progress_label = "{0}%".format(progress_perc)

            if (((i+1) % img_refresh_rate) == 0) or ((i+1) == number_of_iterations):
                som_img = generate_som_image(som)
                set_progress((str(progress_perc), progress_label, som_img))
                time.sleep(0.5)

            store_som_in_cache(session_id, som)

        return 0

    @app.callback(
        [
            Output("learning-progress-bar", "value", allow_duplicate=True),
            Output("som-img", "src", allow_duplicate=True)
        ],
        inputs=Input("stop-learning-btn", "n_clicks"),
        state=[
            State("som-size-slider", "value"),
            State("include-alpha-channel", "value"),
            State("initial-neighbourhood-radius", "value"),
            State("initial-learning-rate", "value"),
            State("neighbourhood-type", "value"),
            State("learning-rate-decay-func", "value"),
            State("red-range-slider", "value"),
            State("green-range-slider", "value"),
            State("blue-range-slider", "value"),
            State("alpha-channel-range-slider", "value"),
            State("session-id", "children"),
        ],
        # running=[
        #     (Output("run-learning-btn", "disabled"), True, False),
        #     (Output("reset-som-btn", "disabled"), True, False),
        #     (Output("stop-learning-btn", "disabled"), True, True),
        #     (Output("som-size-slider", "disabled"), True, False),
        #     (
        #             Output("include-alpha-channel", "options"),
        #             ALPHA_CHANNEL_OPTIONS_DISABLED,
        #             ALPHA_CHANNEL_OPTIONS_ENABLED
        #     ),
        #     (Output("initial-neighbourhood-radius", "disabled"), True, False),
        #     (Output("initial-learning-rate", "disabled"), True, False),
        #     (Output("neighbourhood-type", "disabled"), True, False),
        #     (Output("learning-rate-decay-func", "disabled"), True, False),
        # ],
        # background=True,
        prevent_initial_call=True
    )
    def learning_interrupted(
            n_clicks: int,
            som_size: int,
            include_alpha_channel: bool,
            initial_neighbourhood_radius: int,
            initial_learning_rate: float,
            neighbourhood_type: str,
            learning_rate_decay_func: str,
            red_range: List[int],
            green_range: List[int],
            blue_range: List[int],
            alpha_channel_range: List[int],
            session_id: str
    ):
        """
        When learning process is interrupted we need to reset som and
        write it file - it might happen that learning process is interrupted
        during pickling som object; in such a case when we try to load it afterward
        we might obtain EOFerror. Issue described in the topic:
        https://stackoverflow.com/questions/1653897/if-pickling-was-interrupted-will-unpickling-necessarily-always-fail-python

        :param n_clicks: number of button clicks
        :param som_size: network size
        :param include_alpha_channel: alpha channel indicator
        :param initial_neighbourhood_radius: initial neighbourhood radius as a percent
            of network radius size
        :param initial_learning_rate: initial learning rate
        :param neighbourhood_type: type of neighbourhood function to use - one
            of 'Gaussian' or 'Bubble'
        :param learning_rate_decay_func: type of decay function for learning rate. Possible
            choices are 'Linear', 'Inverse of time' and 'Power series'.
        :param red_range: range of red component
        :param green_range: range of green component
        :param blue_range: range of blue component
        :param alpha_channel_range: alpha channel of red component
        :param session_id: id of current session
        """
        rgba_low = (
            red_range[0],
            green_range[0],
            blue_range[0],
            alpha_channel_range[0]
        )

        rgba_high = (
            red_range[1],
            green_range[1],
            blue_range[1],
            alpha_channel_range[1]
        )

        som = SelfOrganizingMap(
            size=som_size,
            include_alpha_channel=include_alpha_channel,
            initial_neighbourhood_radius=initial_neighbourhood_radius/100,
            initial_learning_rate=initial_learning_rate,
            neighbourhood_type=NeighbourhoodType(neighbourhood_type),
            learning_rate_decay_func=LearningRateDecay(learning_rate_decay_func),
            rgba_low=rgba_low,
            rgba_high=rgba_high
        )
        store_som_in_cache(session_id, som)
        store_img = generate_som_image(som)
        progress_val = 0

        return progress_val, store_img

    @app.callback(
        Output("rgba-range-sliders-collapse", "is_open"),
        Input("show-rgba-range-sliders", "value")
    )
    def rgba_range_sliders_visibility(show_sliders: bool):
        """
        Show or hide Div containing RGBA range sliders

        :param show_sliders: bool - whether to show sliders or not
        """
        return show_sliders
