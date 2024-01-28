"""
Definitions of callbacks functions fot the Dash applications. In order to
make the main app.py file more concise we are going to define callbacks
in this file.
"""

import base64

from dash.dash import Dash
from dash import Input, Output
from SOM.SOM import SelfOrganizingMap
from io import BytesIO

def get_callbacks(app: Dash, som: SelfOrganizingMap) -> None:
    """
    Definitions of callback functions for the Dash application

    :param app: Dash app object
    :param som: SelfOrganizingMap object, representing our Kohonen Network
    """
    @app.callback(
        Output("som-img", "src"),
        [
            Input("som-size-slider", "value"),
            Input("include-alpha-channel", "value")
        ]
    )
    def update_som_image(som_size: int, include_alpha_channel: int):
        """
        Update image representing som values

        :param som_size: size of SOM
        :param include_alpha_channel: bool indicating whether to include alpha channel
        """
        include_alpha_channel = include_alpha_channel == 1
        som.size = som_size
        som.include_alpha_channel = include_alpha_channel

        som_im = som.to_image()
        buff = BytesIO()
        som_im.save(buff, format="PNG")
        im_encoded = 'data:image/png;base64,{}'.format(base64.b64encode(buff.getvalue()).decode("utf-8"))

        return im_encoded
