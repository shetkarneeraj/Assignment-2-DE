from __future__ import annotations

import logging
import threading
from typing import Optional

import dash
from dash import Dash, Input, Output, State, dcc, html
import pandas as pd
import plotly.express as px

from .config import DashboardConfig, MqttConfig, load_config
from .subscriber import MqttSubscriber


logger = logging.getLogger(__name__)


def _build_layout() -> html.Div:
    return html.Div(
        [
            html.H2("Australian Electricity Facilities"),
            html.Div(
                [
                    html.Label("Network Region"),
                    dcc.Dropdown(id="region-filter", placeholder="All regions"),
                ],
                style={"width": "30%", "display": "inline-block", "padding": "0 10px"},
            ),
            html.Div(
                [
                    html.Label("Fuel Technology"),
                    dcc.Dropdown(id="fuel-filter", placeholder="All fuel types"),
                ],
                style={"width": "30%", "display": "inline-block", "padding": "0 10px"},
            ),
            html.Div(
                [
                    html.Label("Metric"),
                    dcc.RadioItems(
                        id="metric-toggle",
                        options=[
                            {"label": "Power (MW)", "value": "power"},
                            {"label": "Emissions (tCO2e)", "value": "emissions"},
                        ],
                        value="power",
                        labelStyle={"display": "inline-block", "margin-right": "10px"},
                    ),
                ]
            ),
            dcc.Graph(id="facility-map"),
            dcc.Interval(id="update-interval", interval=2000, n_intervals=0),
        ]
    )


def create_figure(dataframe, metric: str, mapbox_token: Optional[str]):
    if dataframe.empty:
        return px.scatter_mapbox(
            lat=[],
            lon=[],
            zoom=4,
        )

    latest = (
        dataframe.sort_values("timestamp")
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=["facility_id"], keep="last")
    )
    hover_text = (
        latest["name"].fillna(latest["facility_id"])
        + "<br>"
        + metric.capitalize()
        + ": "
        + latest[metric].round(2).astype(str)
    )
    fig = px.scatter_mapbox(
        latest,
        lat="latitude",
        lon="longitude",
        hover_name="name",
        hover_data={
            "facility_id": True,
            metric: True,
            "fuel_type": True,
            "network_region": True,
        },
        color=metric,
        size=metric,
        zoom=4,
        height=600,
        custom_data=["facility_id", "fuel_type", "network_region", "timestamp"],
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Fuel: %{customdata[1]}<br>Region: %{customdata[2]}<br>Timestamp: %{customdata[3]}<extra></extra>",
        hovertext=hover_text,
    )
    fig.update_layout(
        mapbox_style="open-street-map" if not mapbox_token else "mapbox",
        mapbox_accesstoken=mapbox_token,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return fig


def run_dashboard(
    mqtt_config: MqttConfig,
    dashboard_config: DashboardConfig,
    subscriber: Optional[MqttSubscriber] = None,
) -> Dash:
    subscriber = subscriber or MqttSubscriber(mqtt_config)
    subscriber.start()

    app = dash.Dash(__name__)
    app.layout = _build_layout()

    @app.callback(
        Output("facility-map", "figure"),
        Output("region-filter", "options"),
        Output("fuel-filter", "options"),
        Input("update-interval", "n_intervals"),
        Input("metric-toggle", "value"),
        State("region-filter", "value"),
        State("fuel-filter", "value"),
    )
    def update_map(n_intervals, metric, selected_region, selected_fuel):
        del n_intervals  # Unused
        data = subscriber.store.snapshot()
        if data.empty:
            return create_figure(data, metric, dashboard_config.mapbox_token), [], []

        for column in ["power", "emissions"]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")

        if "metadata" in data.columns:
            metadata_expanded = data["metadata"].apply(lambda m: m or {})
            metadata_df = metadata_expanded.apply(pd.Series)
            data = pd.concat([data.drop(columns=["metadata"]), metadata_df], axis=1)

        if selected_region:
            data = data[data["network_region"] == selected_region]
        if selected_fuel:
            data = data[data["fuel_type"] == selected_fuel]

        regions = sorted(data["network_region"].dropna().unique())
        fuels = sorted(data["fuel_type"].dropna().unique())

        figure = create_figure(data, metric, dashboard_config.mapbox_token)

        region_options = [{"label": region, "value": region} for region in regions]
        fuel_options = [{"label": fuel, "value": fuel} for fuel in fuels]
        return figure, region_options, fuel_options

    def _run():
        logger.info(
            "Starting dashboard at http://%s:%s", dashboard_config.host, dashboard_config.port
        )
        app.run_server(
            host=dashboard_config.host,
            port=dashboard_config.port,
            debug=False,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return app


def main() -> None:
    config = load_config()
    run_dashboard(config.mqtt, config.dashboard)


if __name__ == "__main__":
    main()
