from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import dash
import folium
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html
from folium.plugins import MarkerCluster

from .assignment1_facilities import slugify
from .config import DashboardConfig, MqttConfig, load_config
from .data_processing import load_facility_metadata
from .subscriber import MqttSubscriber


logger = logging.getLogger(__name__)

FUEL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _build_layout() -> html.Div:
    return html.Div(
        [
            html.H1(
                "Australian Electricity Facilities Dashboard",
                style={"textAlign": "center", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Network Region"),
                            dcc.Dropdown(id="region-filter", placeholder="All regions"),
                        ],
                        style={"flex": 1, "marginRight": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label("Fuel Technology"),
                            dcc.Dropdown(id="fuel-filter", placeholder="All fuels"),
                        ],
                        style={"flex": 1, "marginRight": "10px"},
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
                                labelStyle={"display": "inline-block", "marginRight": "10px"},
                            ),
                        ],
                        style={"flex": 1},
                    ),
                ],
                style={"display": "flex", "marginBottom": "15px"},
            ),
            html.Iframe(
                id="facility-map",
                srcDoc="",
                style={"width": "100%", "height": "720px", "border": "none", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
            ),
            html.Div(
                html.P(
                    id="data-status",
                    children="Waiting for live data...",
                    style={"textAlign": "center", "marginTop": "10px", "color": "#555"},
                )
            ),
            dcc.Interval(id="update-interval", interval=2000, n_intervals=0),
        ],
        style={"padding": "20px", "fontFamily": "Arial, sans-serif"},
    )


def _prepare_live_dataframe(live_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    base_metadata = metadata.copy()
    if "name_key" not in base_metadata.columns:
        base_metadata["name_key"] = base_metadata["name"].astype(str).apply(slugify)

    if live_df.empty:
        base_metadata["power"] = pd.NA
        base_metadata["emissions"] = pd.NA
        base_metadata["timestamp"] = pd.NaT
        return base_metadata

    df = live_df.copy()
    if "metadata" in df.columns:
        expanded = df["metadata"].apply(lambda meta: meta or {})
        expanded_df = expanded.apply(pd.Series)
        df = pd.concat([df.drop(columns=["metadata"]), expanded_df], axis=1)

    for column in ["power", "emissions", "price", "demand"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "name" not in df.columns:
        df["name"] = df.get("facility_id")
    df["name_key"] = df["name"].astype(str).apply(slugify)

    latest = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["facility_id", "name_key"], keep="last")
    )

    enriched = base_metadata.merge(
        latest[["facility_id", "name_key", "power", "emissions", "timestamp"]],
        on="facility_id",
        how="left",
        suffixes=("", "_live"),
    )

    missing = enriched["power"].isna() & enriched["name_key"].notna()
    if missing.any():
        fallback = latest.set_index("name_key")
        for column in ["power", "emissions", "timestamp"]:
            if column in fallback.columns:
                enriched.loc[missing, column] = enriched.loc[missing, "name_key"].map(
                    fallback[column]
                )

    return enriched


def _colour_map(fuels: list[str]) -> dict[str, str]:
    return {fuel: FUEL_COLORS[idx % len(FUEL_COLORS)] for idx, fuel in enumerate(sorted(fuels))}


def _build_folium_map(df: pd.DataFrame, metric: str) -> str:
    map_obj = folium.Map(location=[-25.2744, 133.7751], zoom_start=4, tiles="cartodbpositron")
    marker_cluster = MarkerCluster().add_to(map_obj)

    df_vis = df.dropna(subset=["latitude", "longitude"])
    if df_vis.empty:
        return map_obj.get_root().render()

    fuels = [fuel for fuel in df_vis["fuel_type"].dropna().unique()]
    color_map = _colour_map(fuels)
    df_vis["color"] = df_vis["fuel_type"].map(color_map).fillna("#7f7f7f")

    metric_values = df_vis[metric].abs()
    if metric_values.max() > 0:
        size = (metric_values / metric_values.max()) * 10 + 4
    else:
        size = pd.Series([6] * len(df_vis), index=df_vis.index)

    for idx, row in df_vis.iterrows():
        value = row.get(metric)
        value_str = "N/A" if pd.isna(value) else f"{value:.2f}"
        popup_html = f"""
        <b>Name:</b> {row.get('name', row.get('facility_id'))}<br>
        <b>Fuel:</b> {row.get('fuel_type', 'Unknown')}<br>
        <b>Status:</b> {row.get('status', 'N/A')}<br>
        <b>{metric.title()}:</b> {value_str}<br>
        <b>Timestamp:</b> {row.get('timestamp', 'N/A')}
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=float(size.loc[idx]),
            popup=folium.Popup(popup_html, max_width=300),
            color=row["color"],
            fill=True,
            fill_color=row["color"],
            fill_opacity=0.7,
        ).add_to(marker_cluster)

    legend_html = """
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 200px; height: auto;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px;">
    <b>Fuel Types</b><br>
    """
    for fuel, color in color_map.items():
        legend_html += f'<i class="fa fa-circle" style="color:{color}"></i> {fuel}<br>'
    legend_html += "</div>"
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    return map_obj.get_root().render()


def run_dashboard(
    mqtt_config: MqttConfig,
    dashboard_config: DashboardConfig,
    subscriber: Optional[MqttSubscriber] = None,
    metadata_path: Path | None = None,
) -> Dash:
    subscriber = subscriber or MqttSubscriber(mqtt_config)
    subscriber.start()

    metadata_path = metadata_path or Path("data/facilities_metadata.csv")
    facility_metadata = load_facility_metadata(metadata_path)

    app = dash.Dash(__name__)
    app.layout = _build_layout()

    @app.callback(
        Output("facility-map", "srcDoc"),
        Output("region-filter", "options"),
        Output("fuel-filter", "options"),
        Output("data-status", "children"),
        Input("update-interval", "n_intervals"),
        Input("metric-toggle", "value"),
        State("region-filter", "value"),
        State("fuel-filter", "value"),
    )
    def update_map(n_intervals, metric, selected_region, selected_fuel):
        del n_intervals
        live_data = subscriber.store.snapshot()
        combined = _prepare_live_dataframe(live_data, facility_metadata)

        if selected_region:
            combined = combined[combined["network_region"] == selected_region]
        if selected_fuel:
            combined = combined[combined["fuel_type"] == selected_fuel]

        regions = sorted(facility_metadata["network_region"].dropna().unique())
        fuels = sorted(facility_metadata["fuel_type"].dropna().unique())

        map_html = _build_folium_map(combined, metric)
        latest_ts = combined["timestamp"].dropna().max()
        status = (
            f"Displaying {combined.dropna(subset=['latitude', 'longitude']).shape[0]} facilities."
            if pd.isna(latest_ts)
            else f"Last update: {latest_ts}"
        )

        region_options = [{"label": region, "value": region} for region in regions]
        fuel_options = [{"label": fuel, "value": fuel} for fuel in fuels]
        return map_html, region_options, fuel_options, status

    def _run():
        logger.info(
            "Starting dashboard at http://%s:%s",
            dashboard_config.host,
            dashboard_config.port,
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
    run_dashboard(
        config.mqtt,
        config.dashboard,
        metadata_path=config.facilities_metadata_path,
    )


if __name__ == "__main__":
    main()
