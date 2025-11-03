from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import dash
from dash import Dash, Input, Output, State, dcc, html
import pandas as pd
import plotly.express as px

# Suppress pandas fillna deprecation warnings by opting into future behavior
pd.set_option('future.no_silent_downcasting', True)

from .config import DashboardConfig, MqttConfig, load_config
from .data_processing import load_facility_metadata
from .subscriber import MqttSubscriber


logger = logging.getLogger(__name__)


def _build_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "Australian Electricity Facilities Dashboard",
                        style={
                            "textAlign": "center",
                            "color": "#2c3e50",
                            "marginBottom": "20px",
                            "fontSize": "2.5em",
                        },
                    ),
                    html.P(
                        "Real-time visualization of electricity generation and emissions across Australia",
                        style={
                            "textAlign": "center",
                            "color": "#7f8c8d",
                            "marginBottom": "30px",
                            "fontSize": "1.1em",
                        },
                    ),
                ],
                style={"backgroundColor": "#ecf0f1", "padding": "20px", "marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Network Region",
                                style={"fontWeight": "bold", "marginBottom": "5px", "display": "block"},
                            ),
                            dcc.Dropdown(
                                id="region-filter",
                                placeholder="Select a region or 'All'",
                                clearable=True,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block", "padding": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Fuel Technology",
                                style={"fontWeight": "bold", "marginBottom": "5px", "display": "block"},
                            ),
                            dcc.Dropdown(
                                id="fuel-filter",
                                placeholder="Select a fuel type or 'All'",
                                clearable=True,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block", "padding": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Metric",
                                style={"fontWeight": "bold", "marginBottom": "5px", "display": "block"},
                            ),
                            dcc.RadioItems(
                                id="metric-toggle",
                                options=[
                                    {"label": "Power (MW)", "value": "power"},
                                    {"label": "Emissions (tCO2e)", "value": "emissions"},
                                ],
                                value="power",
                                labelStyle={
                                    "display": "inline-block",
                                    "marginRight": "20px",
                                    "marginLeft": "5px",
                                    "fontSize": "1.1em",
                                },
                                inputStyle={"marginRight": "5px"},
                            ),
                        ],
                        style={"width": "35%", "display": "inline-block", "padding": "10px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "marginBottom": "20px",
                    "backgroundColor": "#ffffff",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
            dcc.Graph(
                id="facility-map",
                style={"height": "700px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
            ),
            html.Div(
                [
                    html.P(
                        id="data-status",
                        children="Waiting for data...",
                        style={
                            "textAlign": "center",
                            "color": "#7f8c8d",
                            "marginTop": "15px",
                            "fontSize": "0.9em",
                        },
                    ),
                ],
            ),
            dcc.Interval(id="update-interval", interval=2000, n_intervals=0),
        ],
        style={"padding": "20px", "fontFamily": "Arial, sans-serif", "backgroundColor": "#f8f9fa"},
    )


def create_figure(dataframe, metric: str, mapbox_token: Optional[str]):
    if dataframe.empty:
        # Use scatter_map instead of deprecated scatter_mapbox
        return px.scatter_map(
            lat=[],
            lon=[],
            zoom=4,
            map_style="open-street-map",
        )

    latest = (
        dataframe.sort_values("timestamp")
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=["facility_id"], keep="last")
    )
    # Create name column with fallback to facility_id, avoiding fillna deprecation
    name_col = latest["name"].copy()
    mask = name_col.isna() | (name_col == "")
    name_col.loc[mask] = latest.loc[mask, "facility_id"]
    
    hover_text = (
        name_col.astype(str)
        + "<br>"
        + metric.capitalize()
        + ": "
        + latest[metric].round(2).astype(str)
    )
    # Use absolute values for size (can't be negative)
    size_column = latest[metric].abs()
    # Normalize size to a reasonable range (0-30)
    if size_column.max() > 0:
        size_column = size_column / size_column.max() * 30
    size_column = size_column.clip(lower=3)  # Minimum size of 3
    
    # Use scatter_map instead of deprecated scatter_mapbox
    fig = px.scatter_map(
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
        size=size_column,
        zoom=4,
        height=700,
        custom_data=["facility_id", "fuel_type", "network_region", "timestamp", metric],
        color_continuous_scale="Viridis",
        size_max=30,
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      f"{metric.capitalize()}: %{{customdata[4]:.2f}}<br>" +
                      "Fuel: %{customdata[1]}<br>" +
                      "Region: %{customdata[2]}<br>" +
                      "Timestamp: %{customdata[3]}<extra></extra>",
        hovertext=hover_text,
    )
    map_style = "open-street-map" if not mapbox_token else "mapbox"
    fig.update_layout(
        map_style=map_style,
        mapbox_accesstoken=mapbox_token if mapbox_token else None,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_colorbar=dict(title=metric.capitalize()),
    )
    return fig


def run_dashboard(
    mqtt_config: MqttConfig,
    dashboard_config: DashboardConfig,
    subscriber: Optional[MqttSubscriber] = None,
) -> Dash:
    subscriber = subscriber or MqttSubscriber(mqtt_config)
    subscriber.start()

    # Load facility metadata for lat/lon if available
    facility_metadata = None
    try:
        config = load_config()
        facility_metadata = load_facility_metadata(config.facilities_metadata_path)
        logger.info(f"Loaded facility metadata: {len(facility_metadata)} facilities")
    except FileNotFoundError:
        logger.warning("Facility metadata file not found. Attempting to fetch from API...")
        # Try to get coordinates from facilities API
        try:
            from .api_client import OpenElectricityClient
            # Use the same config that was loaded
            api_client = OpenElectricityClient(config.api)
            # The fetch_facilities method should use the same authenticated session
            facilities = api_client.fetch_facilities(config.api.network)
            
            # Extract lat/lon from facilities API response
            facility_dicts = []
            for fac in facilities:
                fac_dict = {"facility_id": fac.get("code"), "name": fac.get("name")}
                
                # Try to extract location
                location = fac.get("location")
                if isinstance(location, dict):
                    fac_dict["latitude"] = location.get("latitude") or location.get("lat")
                    fac_dict["longitude"] = location.get("longitude") or location.get("lon") or location.get("lng")
                elif isinstance(location, str):
                    # Try to parse if it's a JSON string
                    import json
                    try:
                        loc_data = json.loads(location)
                        fac_dict["latitude"] = loc_data.get("latitude") or loc_data.get("lat")
                        fac_dict["longitude"] = loc_data.get("longitude") or loc_data.get("lon") or loc_data.get("lng")
                    except:
                        pass
                
                fac_dict["network_region"] = fac.get("network_region")
                
                # Extract fuel_type from units array
                units = fac.get("units", [])
                fuel_types = set()
                if isinstance(units, list):
                    for unit in units:
                        fueltech = unit.get("fueltech_id") or unit.get("fuel_type")
                        if fueltech:
                            fuel_types.add(fueltech)
                # Use the first fuel type or join if multiple
                if fuel_types:
                    fac_dict["fuel_type"] = ", ".join(sorted(fuel_types)) if len(fuel_types) > 1 else list(fuel_types)[0]
                else:
                    fac_dict["fuel_type"] = None
                
                facility_dicts.append(fac_dict)
            
            if facility_dicts:
                facility_metadata = pd.DataFrame(facility_dicts)
                logger.info(f"Fetched {len(facility_metadata)} facilities from API")
        except Exception as api_error:
            logger.warning(f"Could not fetch facilities from API: {api_error}")
    except Exception as e:
        logger.warning(f"Could not load facility metadata: {e}")

    app = dash.Dash(__name__)
    app.layout = _build_layout()
    
    # Store facility_metadata in a way accessible to the callback
    # Use a closure variable that's captured by the callback function

    @app.callback(
        Output("facility-map", "figure"),
        Output("region-filter", "options"),
        Output("fuel-filter", "options"),
        Output("data-status", "children"),
        Input("update-interval", "n_intervals"),
        Input("metric-toggle", "value"),
        State("region-filter", "value"),
        State("fuel-filter", "value"),
    )
    def update_map(n_intervals, metric, selected_region, selected_fuel):
        # Log periodically to show updates are happening
        if n_intervals % 15 == 0:  # Every 30 seconds (15 * 2 second interval)
            data_count = len(subscriber.store.snapshot())
            logger.info(f"Dashboard update #{n_intervals}: {data_count} messages received")
        
        data = subscriber.store.snapshot()
        if data.empty:
            logger.debug("No data available yet - showing empty map")
            status = "⏳ Waiting for MQTT messages..."
            return create_figure(data, metric, dashboard_config.mapbox_token), [], [], status

        # Convert numeric columns
        for column in ["power", "emissions"]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")

        # Extract metadata properly - handle both dict and string formats
        if "metadata" in data.columns:
            import json
            def extract_metadata(m):
                if pd.isna(m):
                    return {}
                if isinstance(m, dict):
                    return m
                if isinstance(m, str):
                    try:
                        return json.loads(m)
                    except:
                        return {}
                return {}
            
            metadata_expanded = data["metadata"].apply(extract_metadata)
            metadata_df = pd.json_normalize(metadata_expanded)
            
            # Merge metadata columns into main dataframe
            for col in metadata_df.columns:
                if col not in data.columns:
                    data[col] = metadata_df[col]
            
            # Remove the metadata column after extraction
            data = data.drop(columns=["metadata"], errors="ignore")
        
        # Merge facility metadata if available (for lat/lon)
        if facility_metadata is not None and "facility_id" in data.columns:
            # Merge on facility_id to get lat/lon from metadata file
            metadata_cols = ["facility_id"]
            for col in ["latitude", "longitude", "name", "fuel_type", "network_region"]:
                if col in facility_metadata.columns:
                    metadata_cols.append(col)
            
            data = data.merge(
                facility_metadata[metadata_cols],
                on="facility_id",
                how="left",
                suffixes=("", "_metadata")
            )
            # Use metadata file values if original values are null
            if "latitude_metadata" in data.columns:
                data["latitude"] = data["latitude"].fillna(data["latitude_metadata"]).infer_objects(copy=False)
                data["longitude"] = data["longitude"].fillna(data["longitude_metadata"]).infer_objects(copy=False)
                data = data.drop(columns=["latitude_metadata", "longitude_metadata"], errors="ignore")
            if "name_metadata" in data.columns:
                data["name"] = data["name"].fillna(data["name_metadata"]).infer_objects(copy=False)
                data = data.drop(columns=["name_metadata"], errors="ignore")
            if "fuel_type_metadata" in data.columns:
                # Use metadata file values, but keep non-null original values
                mask = data["fuel_type"].isna() | (data["fuel_type"] == None) | (data["fuel_type"] == "")
                data.loc[mask, "fuel_type"] = data.loc[mask, "fuel_type_metadata"]
                data = data.drop(columns=["fuel_type_metadata"], errors="ignore")
            if "network_region_metadata" in data.columns:
                # Use metadata file values, but keep non-null original values
                mask = data["network_region"].isna() | (data["network_region"] == None) | (data["network_region"] == "")
                data.loc[mask, "network_region"] = data.loc[mask, "network_region_metadata"]
                data = data.drop(columns=["network_region_metadata"], errors="ignore")

        # Ensure we have network_region and fuel_type columns
        if "network_region" not in data.columns:
            data["network_region"] = None
        if "fuel_type" not in data.columns:
            data["fuel_type"] = None

        # Apply filters
        if selected_region:
            data = data[data["network_region"] == selected_region]
        if selected_fuel:
            data = data[data["fuel_type"] == selected_fuel]

        # Get unique values for dropdowns ONLY from actual MQTT data (not from facility_metadata)
        # This ensures only technologies/regions present in the data are shown
        all_data = subscriber.store.snapshot()
        if "metadata" in all_data.columns:
            all_metadata = all_data["metadata"].apply(
                lambda m: m if isinstance(m, dict) else (json.loads(m) if isinstance(m, str) else {})
            )
            all_metadata_df = pd.json_normalize(all_metadata)
            for col in all_metadata_df.columns:
                if col not in all_data.columns:
                    all_data[col] = all_metadata_df[col]
        
        # Extract regions and fuels ONLY from parsed MQTT data
        # Merge facility metadata to enrich MQTT data with region/fuel info if missing
        if facility_metadata is not None and "facility_id" in all_data.columns:
            metadata_cols = ["facility_id"]
            for col in ["name", "fuel_type", "network_region"]:
                if col in facility_metadata.columns:
                    metadata_cols.append(col)
            
            all_data = all_data.merge(
                facility_metadata[metadata_cols],
                on="facility_id",
                how="left",
                suffixes=("", "_metadata")
            )
            
            # Fill in missing values from metadata
            if "fuel_type_metadata" in all_data.columns:
                mask = all_data["fuel_type"].isna() | (all_data["fuel_type"] == "") | (all_data["fuel_type"] == None)
                all_data.loc[mask, "fuel_type"] = all_data.loc[mask, "fuel_type_metadata"]
                all_data = all_data.drop(columns=["fuel_type_metadata"], errors="ignore")
            if "network_region_metadata" in all_data.columns:
                mask = all_data["network_region"].isna() | (all_data["network_region"] == "") | (all_data["network_region"] == None)
                all_data.loc[mask, "network_region"] = all_data.loc[mask, "network_region_metadata"]
                all_data = all_data.drop(columns=["network_region_metadata"], errors="ignore")
        
        # Get unique values ONLY from data that actually exists in MQTT messages
        regions = []
        fuels = []
        
        # Try multiple column names for regions
        region_col = None
        for col in ["network_region", "region"]:
            if col in all_data.columns:
                region_col = col
                break
        
        # Try multiple column names for fuels
        fuel_col = None
        for col in ["fuel_type"]:
            if col in all_data.columns:
                fuel_col = col
                break
        
        if region_col and not all_data.empty:
            regions = sorted([r for r in all_data[region_col].dropna().unique().tolist() 
                             if r is not None and str(r) != 'nan' and str(r).strip() != '' and r != ''])
        if fuel_col and not all_data.empty:
            fuels = sorted([f for f in all_data[fuel_col].dropna().unique().tolist() 
                           if f is not None and str(f) != 'nan' and str(f).strip() != '' and f != ''])
        
        # Log for debugging
        if n_intervals % 30 == 0:  # Every 60 seconds
            logger.info(f"Dropdown options from MQTT data - Regions: {len(regions)}, Fuels: {len(fuels)}")
            if regions:
                logger.debug(f"Sample regions: {regions[:5]}")
            if fuels:
                logger.debug(f"Sample fuels: {fuels[:5]}")

        # Create figure
        figure = create_figure(data, metric, dashboard_config.mapbox_token)

        # Create dropdown options
        region_options = [{"label": "All Regions", "value": None}] + [
            {"label": region, "value": region} for region in regions if region
        ]
        fuel_options = [{"label": "All Fuel Types", "value": None}] + [
            {"label": fuel, "value": fuel} for fuel in fuels if fuel
        ]

        # Status message
        facility_count = len(data.drop_duplicates(subset=["facility_id"])) if not data.empty else 0
        total_facilities = len(all_data.drop_duplicates(subset=["facility_id"])) if not all_data.empty else 0
        
        # Count facilities with coordinates
        if not data.empty and "latitude" in data.columns and "longitude" in data.columns:
            facilities_with_coords = len(data.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=["facility_id"]))
            if facilities_with_coords > 0:
                status = f"📍 Showing {facility_count} facilities ({total_facilities} total) | {facilities_with_coords} with coordinates | {len(data)} data points"
            else:
                status = f"📍 Showing {facility_count} facilities ({total_facilities} total) | {len(data)} data points | ⚠️ No coordinates available - facility metadata file missing"
        else:
            status = f"📍 Showing {facility_count} facilities ({total_facilities} total) | {len(data)} data points | ⚠️ No coordinates available"

        return figure, region_options, fuel_options, status

    def _run():
        try:
            logger.info(
                "Starting dashboard at http://%s:%s", dashboard_config.host, dashboard_config.port
            )
            print(f"\n{'='*60}")
            print(f"Dashboard is running at: http://{dashboard_config.host}:{dashboard_config.port}")
            print(f"{'='*60}\n")
            app.run(
                host=dashboard_config.host,
                port=dashboard_config.port,
                debug=False,
            )
        except Exception as e:
            logger.error("Error running dashboard: %s", e, exc_info=True)
            print(f"ERROR: Dashboard failed to start: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Dashboard thread started. Waiting for server to initialize...")
    time.sleep(2)  # Give the server a moment to start
    return app


def main() -> None:
    config = load_config()
    app = run_dashboard(config.mqtt, config.dashboard)
    # Keep the main thread alive so the dashboard thread can run
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard...")


if __name__ == "__main__":
    main()
