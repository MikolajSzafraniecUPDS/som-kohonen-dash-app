"""
Definitions of callbacks functions fot the Dash applications. In order to
make the main app.py file more concise we are going to define callbacks
in this file.
"""

import base64

from dash.dash import Dash
from dash import Input, Output, State
from SOM.SOM import SelfOrganizingMap, NeighbourhoodType, LearningRateDecay
from io import BytesIO
from app_components.utils import get_som_from_cache, write_som_to_cache, rm_som_from_cache


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
            Input("learning-rate-decay-func", "value")
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
            session_id: str
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
        :param session_id: id of current session
        """
        som = get_som_from_cache(session_id)
        size_same = som_size == som.size
        alpha_channel_same = include_alpha_channel == som.include_alpha_channel
        neighbourhood_radius_same = (initial_neighbourhood_radius/100) == som.initial_neighbourhood_radius
        learning_rate_same = initial_learning_rate == som.initial_learning_rate
        neighbourhood_type_same = neighbourhood_type == som.neighbourhood_type.value
        learning_rate_decay_func_same = learning_rate_decay_func == som.learning_rate_decay_func.value

        update_button_disabled = all(
            [size_same, alpha_channel_same, neighbourhood_radius_same,
             learning_rate_same, neighbourhood_type_same, learning_rate_decay_func_same]
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
            State("session-id", "children")
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
            session_id: str
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
        :param session_id: id of current session
        """
        som = get_som_from_cache(session_id)
        current_size = som.size
        current_alpha_channel_indicator = som.include_alpha_channel

        # If both size and alpha channel indicator are going to be updated we
        # don't want to reset network twice
        if (current_size != som_size) and (current_alpha_channel_indicator != include_alpha_channel):
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

        som_img = generate_som_image(som)
        write_som_to_cache(session_id, som)
        return som_img,

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
            Output("update-network-btn", "disabled", allow_duplicate=True)
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

        return (som_size, som_alpha_channel_indicator,
                som_initial_neighbourhood_radius, som_initial_learning_rate,
                som_neighbourhood_type, som_learning_rate_decay_func,
                True)

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
        write_som_to_cache(session_id, som)
        som_img = generate_som_image(som)
        return som_img

    # @app.callback(
    #     Output("app-closed", "children"),
    #     inputs=Input("clear_cache_btn", "n_clicks"),
    #     state=State("session-id", "children")
    # )
    # def clear_cache(n_clicks: int, session_id: str):
    #     """
    #     Remove som object from cache when session is ended
    #
    #     :param n_clicks: how many times reset button was clicked
    #     :param session_id: id of current session
    #     """
    #     rm_som_from_cache(session_id)
    #     return "File removed"
