"""
Dash application
"""
import os
import dash_bootstrap_components as dbc

from dash import Dash, DiskcacheManager, CeleryManager, Input, Output, html, callback
from SOM.SOM import SelfOrganizingMap
from app_components.tabs_components import *

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
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

# Initialize the app
external_stylesheets = [dbc.themes.DARKLY]
app = Dash(
    "som_rgb_example",
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True  # We generate tab content dynamically, so this flag must be set as True
)

# Initialize instance of Self-Organising map
som = SelfOrganizingMap()

# App layout
app.layout = html.Div([
    dbc.Tabs(
        id="section-selection",
        active_tab="som-setup-and-results",
        children=[
            dbc.Tab(label="SOM setup and results", tab_id="som-setup-and-results")
        ]
    )
])

# Define a way of updating tabs of dashboard
@callback(
    Output("output-tab", "children"),
    Input("section-selection", "active_tab")
)
def render_tab_content(tab_name: str) -> html.Div:
    """
    Render tab content dynamically. Such an approach is recommended
    due to the fact, that otherwise content for all tabs would be
    generated at the same moment, which could cause a performance
    issues.

    :param tab_name: id of tab to show
    """
    if tab_name == "commits-timeline":
        return render_commits_timeline_div()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
