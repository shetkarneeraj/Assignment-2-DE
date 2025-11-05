from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dash
import folium
import pandas as pd
from dash import Dash, Input, Output, dcc, html
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
    return html.Div([
        html.Div([
            html.H1("Electricity Facilities", style={
                "textAlign": "center", "margin": "20px 0", "fontSize": "2.5em",
                "fontWeight": "300", "color": "#2c3e50", "letterSpacing": "2px"
            }),
            html.Div([
                dcc.RadioItems(
                    id="metric-toggle",
                    options=[
                        {"label": "Power (MW)", "value": "power"},
                        {"label": "Emissions (tCO2e)", "value": "emissions"}
                    ],
                    value="power",
                    style={"textAlign": "center", "marginBottom": "20px"},
                    labelStyle={"margin": "0 20px", "fontSize": "1.1em", "color": "#34495e"}
                )
            ]),
            html.Iframe(
                id="facility-map",
                srcDoc="",
                style={
                    "width": "100%", "height": "80vh", "border": "none",
                    "borderRadius": "8px", "boxShadow": "0 4px 20px rgba(0,0,0,0.1)"
                }
            ),
            html.Div([
                html.P(id="data-status", style={
                    "textAlign": "center", "margin": "20px 0", "fontSize": "1.1em",
                    "color": "#7f8c8d", "fontStyle": "italic"
                })
            ]),
            dcc.Interval(id="update-interval", interval=3000, n_intervals=0)
        ], style={
            "maxWidth": "1400px", "margin": "0 auto", "padding": "20px",
            "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
            "backgroundColor": "#f8f9fa", "minHeight": "100vh"
        })
    ], style={"backgroundColor": "#f8f9fa"})


def _prepare_live_dataframe(live_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    meta = metadata.copy()
    if "name_key" not in meta.columns:
        meta["name_key"] = meta["name"].astype(str).apply(slugify)

    # If no live data, return metadata with empty power/emissions/timestamp
    if live_df.empty:
        meta["power"] = pd.NA
        meta["emissions"] = pd.NA
        meta["timestamp"] = pd.NaT
        return meta

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

    meta = metadata.copy()
    if "name_key" not in meta.columns:
        meta["name_key"] = meta["name"].astype(str).apply(slugify)

    meta_columns = [
        "facility_id",
        "name",
        "name_key",
        "fuel_type",
        "network_region",
        "latitude",
        "longitude",
        "status",
    ]
    available_meta_cols = [col for col in meta_columns if col in meta.columns]

    enriched = latest.merge(
        meta[available_meta_cols],
        on="facility_id",
        how="left",
        suffixes=("", "_meta"),
    )

    # Ensure coordinate columns have proper numeric dtype from the start
    for coord_col in ["latitude", "longitude"]:
        if coord_col in enriched.columns:
            enriched[coord_col] = pd.to_numeric(enriched[coord_col], errors='coerce')
        meta_coord = f"{coord_col}_meta"
        if meta_coord in enriched.columns:
            enriched[meta_coord] = pd.to_numeric(enriched[meta_coord], errors='coerce')

    # Fill missing metadata from the merged columns using dtype-safe where-based assignment
    for column in ["name", "fuel_type", "network_region", "latitude", "longitude"]:
        meta_col = f"{column}_meta"
        if column in enriched.columns and meta_col in enriched.columns:
            if column in ["latitude", "longitude"]:
                base = pd.to_numeric(enriched[column], errors='coerce')
                meta_vals = pd.to_numeric(enriched[meta_col], errors='coerce')
                enriched[column] = base.where(base.notna(), meta_vals).astype('float64')
            else:
                base = enriched[column]
                meta_vals = enriched[meta_col]
                enriched[column] = base.where(base.notna(), meta_vals)
                if enriched[column].dtype == 'object':
                    enriched[column] = enriched[column].infer_objects(copy=False)
    
    # For facilities that didn't match by facility_id, try matching by name_key
    if "name_key" in enriched.columns and "latitude" in enriched.columns:
        missing_coords = enriched["latitude"].isna()
        if missing_coords.any():
            name_lookup = meta.drop_duplicates(subset=["name_key"]).set_index("name_key")

            # Ensure lookup columns have proper dtypes
            for coord_col in ["latitude", "longitude"]:
                if coord_col in name_lookup.columns:
                    name_lookup[coord_col] = pd.to_numeric(name_lookup[coord_col], errors='coerce')

            for column in ["name", "fuel_type", "network_region", "latitude", "longitude"]:
                if column in name_lookup.columns and column in enriched.columns:
                    # Map by name_key for missing values
                    missing_mask = enriched[column].isna() & enriched["name_key"].notna()
                    if missing_mask.any():
                        mapped_values = enriched.loc[missing_mask, "name_key"].map(name_lookup[column])
                        # Use where-based replacement to avoid dtype conflicts
                        if column in ["latitude", "longitude"]:
                            current = pd.to_numeric(enriched[column], errors='coerce')
                            replacements = pd.to_numeric(mapped_values, errors='coerce')
                            combined = current.where(~missing_mask, replacements)
                            enriched[column] = combined.astype('float64')
                        else:
                            current = enriched[column]
                            combined = current.where(~missing_mask, mapped_values)
                            enriched[column] = combined

    # Infer object types to suppress pandas warning
    for column in ["name", "fuel_type", "network_region"]:
        if column in enriched.columns:
            enriched[column] = enriched[column].infer_objects(copy=False)

    return enriched


def _colour_map(fuels: list[str]) -> dict[str, str]:
    return {fuel: FUEL_COLORS[idx % len(FUEL_COLORS)] for idx, fuel in enumerate(sorted(fuels))}


def _build_folium_map(df: pd.DataFrame, metric: str) -> str:
    map_obj = folium.Map(location=[-25.2744, 133.7751], zoom_start=4, tiles="cartodbpositron")
    marker_cluster = MarkerCluster().add_to(map_obj)

    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return map_obj.get_root().render()

    df_vis = df.dropna(subset=["latitude", "longitude"])
    if df_vis.empty:
        return map_obj.get_root().render()

    fuels = [fuel for fuel in df_vis["fuel_type"].dropna().unique()]
    color_map = _colour_map(fuels)
    df_vis = df_vis.copy()  # Avoid SettingWithCopyWarning
    df_vis["color"] = df_vis["fuel_type"].map(color_map).fillna("#7f7f7f")

    metric_values = df_vis[metric].abs()
    if metric_values.max() > 0:
        size = (metric_values / metric_values.max()) * 10 + 4
    else:
        size = pd.Series([6] * len(df_vis), index=df_vis.index)

    for idx, row in df_vis.iterrows():
        value = row.get(metric)
        value_str = "N/A" if pd.isna(value) else f"{value:.2f}"

        power_value = row.get("power")
        power_str = "N/A" if pd.isna(power_value) else f"{power_value:.2f} MW"

        emissions_value = row.get("emissions")
        emissions_str = "N/A" if pd.isna(emissions_value) else f"{emissions_value:.2f} tCO2e"

        timestamp = row.get("timestamp")
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(timestamp) and hasattr(timestamp, "strftime") else "N/A"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif;">
        <b>Station:</b> {row.get('name', row.get('facility_id', 'Unknown'))}<br>
        <b>Type:</b> {row.get('fuel_type', 'Unknown')}<br>
        <hr style="margin: 5px 0;">
        <b>Power:</b> {power_str}<br>
        <b>Emissions:</b> {emissions_str}<br>
        <b>Region:</b> {row.get('network_region', 'N/A')}<br>
        <b>Updated:</b> {timestamp_str}
        </div>"""

        tooltip_text = f"{row.get('name', row.get('facility_id', 'Unknown'))}: {value_str}"

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=float(size.loc[idx]),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip_text,
            color=row["color"],
            fill=True,
            fill_color=row["color"],
            fill_opacity=0.7,
        ).add_to(marker_cluster)

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; right: 20px; background: rgba(255,255,255,0.95);
                padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                font-family: 'Segoe UI', sans-serif; font-size: 13px; max-width: 200px;">
    <div style="font-weight: 600; margin-bottom: 10px; color: #2c3e50;">Fuel Types</div>"""

    for fuel, color in color_map.items():
        legend_html += f'<div style="margin: 4px 0;"><span style="color:{color}; font-size: 12px;">●</span> {fuel}</div>'
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
        Output("data-status", "children"),
        Input("update-interval", "n_intervals"),
        Input("metric-toggle", "value")
    )
    def update_map(n_intervals, metric):
        del n_intervals
        live_data = subscriber.store.snapshot()
        combined = _prepare_live_dataframe(live_data, facility_metadata)

        map_html = _build_folium_map(combined, metric)
        if combined.empty or "timestamp" not in combined.columns:
            status = "Waiting for live data..."
        else:
            latest_ts = combined["timestamp"].dropna().max()
            display_count = (
                combined.dropna(subset=["latitude", "longitude"]).shape[0]
                if {"latitude", "longitude"}.issubset(combined.columns)
                else len(combined)
            )
            status = (
                "Waiting for live data..."
                if pd.isna(latest_ts)
                else f"Last update: {latest_ts.strftime('%Y-%m-%d %H:%M:%S')} · {display_count} facilities"
            )

        return map_html, status

    app.run(
        host=dashboard_config.host,
        port=dashboard_config.port,
        debug=False,
        use_reloader=False,
    )
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
