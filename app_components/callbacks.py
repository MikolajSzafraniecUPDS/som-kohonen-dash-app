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
            Output("run-learning-btn", "disabled", allow_duplicate=True)
        ],
        [
            Input("som-size-slider", "value"),
            Input("include-alpha-channel", "value")
        ],
        prevent_initial_call=True
    )
    def learning_updating_buttons_status(som_size: int, include_alpha_channel: bool):
        """
        Check whether are there any changes in som parameters and
        enable update button if there are

        :param som_size: network size
        :param include_alpha_channel: alpha channel indicator
        """
        size_same = som_size == som.size
        alpha_channel_same = include_alpha_channel == som.include_alpha_channel

        update_button_disabled = all([size_same, alpha_channel_same])
        run_learning_button_disabled = not update_button_disabled

        return update_button_disabled, run_learning_button_disabled

    @app.callback(
        [
            Output("som-img", "src"),
            Output("update-network-btn", "disabled", allow_duplicate=True),
            Output("run-learning-btn", "disabled", allow_duplicate=True)
        ],
        inputs=Input("update-network-btn", "n_clicks"),
        state=[
            State("som-size-slider", "value"),
            State("include-alpha-channel", "value")
        ],
        prevent_initial_call=True
    )
    def update_network(n_clicks: int, som_size: int, include_alpha_channel: bool):
        """
        Update network and print its image representation in app

        :param n_clicks: number of button clicks
        :param som_size: size of network
        :param include_alpha_channel: value indicating whether alpha channel should be
            included
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

        som_img = generate_som_image(som)
        return som_img, True, False
