"""
Dash application
"""
import os
import uuid

from dash import Dash, DiskcacheManager, CeleryManager, Input, Output, State
from app_components.tabs_components import *
from app_components.callbacks import get_callbacks
from app_components.utils import CACHE_DIR, store_som_in_cache, get_som_from_cache

# Set background callback manager (required to dynamically change Outputs during the
# processing - in our case process of learning the network). More details in documentation:
# https://dash.plotly.com/background-callbacks
if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)
else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache(CACHE_DIR)
    background_callback_manager = DiskcacheManager(cache)

# Initialize the app
external_stylesheets = [dbc.themes.DARKLY]
app = Dash(
    "som_rgb_example",
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,  # We generate tab content dynamically, so this flag must be set as True
    background_callback_manager=background_callback_manager
)


def serve_layout():
    session_id = str(uuid.uuid4())

    res = html.Div([
        html.Div(session_id, id="session-id", style={"visibility": "hidden"}),
        html.H1("RGB Self Organizing Map"),
        html.Br(),
        dbc.Tabs(
            id="section-selection",
            active_tab="som-setup-and-results",
            children=[
                dbc.Tab(label="SOM learning", tab_id="som-setup-and-results"),
                dbc.Tab(label="Learning parameters - explanation", tab_id="about-learning-params")
            ]
        ),
        html.Div(id="output-tab"),
        html.Div([
            dbc.Button(id="clear_cache_btn"),
            html.Div(id="app-closed")
        ], style={"display": "none"})
    ])

    som = SelfOrganizingMap()
    store_som_in_cache(session_id, som)

    return res


# App layout
app.layout = serve_layout


# Define a way of updating tabs of dashboard
@app.callback(
    Output("output-tab", "children"),
    inputs=Input("section-selection", "active_tab"),
    state=State("session-id", "children")
)
def render_tab_content(tab_id: str, session_id: str) -> html.Div:
    """
    Render tab content dynamically. Such an approach is recommended
    due to the fact, that otherwise content for all tabs would be
    generated at the same moment, which could cause a performance
    issues.

    :param tab_id: id of tab to show
    :param session_id: session id
    """
    if tab_id == "som-setup-and-results":
        som = get_som_from_cache(session_id)
        return render_som_setup_and_results_div(som)
    if tab_id == "about-learning-params":
        return render_about_learning_params_tab()


get_callbacks(app)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
