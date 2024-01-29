"""
Definitions of callbacks functions fot the Dash applications. In order to
make the main app.py file more concise we are going to define callbacks
in this file.
"""

import base64

from dash.dash import Dash
from dash import Input, Output, State
from SOM.SOM import SelfOrganizingMap
from io import BytesIO


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


def get_callbacks(app: Dash, som: SelfOrganizingMap) -> None:
    """
    Definitions of callback functions for the Dash application

    :param app: Dash app object
    :param som: SelfOrganizingMap object, representing Kohonen Network
    """

    @app.callback(
        [
            Output("update-network-btn", "disabled", allow_duplicate=True),
            Output("reset-settings-changes-btn", "disabled", allow_duplicate=True),
            Output("run-learning-btn", "disabled", allow_duplicate=True),
            Output("reset-som-btn", "disabled", allow_duplicate=True)
        ],
        [
            Input("som-size-slider", "value"),
            Input("include-alpha-channel", "value"),
            Input("initial-neighbourhood-radius", "value")
        ],
        prevent_initial_call=True
    )
    def learning_updating_buttons_status(
            som_size: int,
            include_alpha_channel: bool,
            initial_neighbourhood_radius: int
    ):
        """
        Check whether are there any changes in som parameters and
        enable update button if there are

        :param som_size: network size
        :param include_alpha_channel: alpha channel indicator
        :param initial_neighbourhood_radius: initial neighbourhood radius as a percent
            of network radius size
        """
        size_same = som_size == som.size
        alpha_channel_same = include_alpha_channel == som.include_alpha_channel
        neighbourhood_radius_same = (initial_neighbourhood_radius/100) == som.initial_neighbourhood_radius

        settings_buttons_disabled = all([size_same, alpha_channel_same, neighbourhood_radius_same])
        learning_reset_buttons_disabled = not settings_buttons_disabled

        return (settings_buttons_disabled, settings_buttons_disabled,
                learning_reset_buttons_disabled, learning_reset_buttons_disabled)

    @app.callback(
        [
            Output("som-img", "src", allow_duplicate=True),
            Output("update-network-btn", "disabled", allow_duplicate=True),
            Output("reset-settings-changes-btn", "disabled", allow_duplicate=True),
            Output("run-learning-btn", "disabled", allow_duplicate=True),
            Output("reset-som-btn", "disabled", allow_duplicate=True)
        ],
        inputs=Input("update-network-btn", "n_clicks"),
        state=[
            State("som-size-slider", "value"),
            State("include-alpha-channel", "value"),
            State("initial-neighbourhood-radius", "value")
        ],
        prevent_initial_call=True
    )
    def update_network(
            n_clicks: int,
            som_size: int,
            include_alpha_channel: bool,
            initial_neighbourhood_radius: int
    ):
        """
        Update network and print its image representation in app

        :param n_clicks: number of button clicks
        :param som_size: size of network
        :param include_alpha_channel: value indicating whether alpha channel should be
            included
        :param initial_neighbourhood_radius: initial neighbourhood radius as a percent
            of network radius size
        """
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

        som_img = generate_som_image(som)
        return som_img, True, True, False, False

    @app.callback(
        [
            Output("som-size-slider", "value"),
            Output("include-alpha-channel", "value"),
            Output("initial-neighbourhood-radius", "value"),
            Output("update-network-btn", "disabled", allow_duplicate=True),
            Output("reset-settings-changes-btn", "disabled", allow_duplicate=True),
        ],
        Input("reset-settings-changes-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def reset_settings_changes(n_clicks: int):
        """
        Reset changes of network settings

        :param n_clicks: how many times reset button was clicked
        """
        som_size = som.size
        som_alpha_channel_indicator = som.include_alpha_channel
        som_initial_neighbourhood_radius = som.initial_neighbourhood_radius*100

        return (som_size, som_alpha_channel_indicator,
                som_initial_neighbourhood_radius, True, True)

    @app.callback(
        Output("som-img", "src", allow_duplicate=True),
        Input("reset-som-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def reset_network(n_clicks: int):
        """
        Reset network after learning procedure

        :param n_clicks: how many times reset button was clicked
        """
        som.reset_network()
        som_img = generate_som_image(som)
        return som_img

