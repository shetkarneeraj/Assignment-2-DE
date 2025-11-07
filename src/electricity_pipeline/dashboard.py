from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dash
import folium
import pandas as pd
from dash import Dash, Input, Output, dcc, html, clientside_callback
from flask import jsonify
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
            html.Div(
                id="facility-map",
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
            # Store for current data
            dcc.Store(id="live-data-store", data=[]),
            # Hidden divs for clientside communication
            html.Div(id="map-update-trigger", style={"display": "none"}),
            html.Div(id="status-update-trigger", style={"display": "none"}),
        ], style={
            "maxWidth": "1400px", "margin": "0 auto", "padding": "20px",
            "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
            "backgroundColor": "#f8f9fa", "minHeight": "100vh"
        }),
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

    # Initialize Dash with custom routes_pathname_prefix to avoid conflicts
    app = dash.Dash(__name__, routes_pathname_prefix='/dash/', suppress_callback_exceptions=True)

    # Create a simple HTML page that works without Dash complications
    @app.server.route('/', methods=['GET'])
    def index():
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Electricity Facilities Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        #facility-map { height: 80vh; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .status { text-align: center; margin: 20px 0; font-size: 1.1em; color: #7f8c8d; }
        .controls { text-align: center; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Electricity Facilities Dashboard</h1>

        <div class="controls">
            <label><input type="radio" name="metric" value="power" checked> Power (MW)</label>
            <label style="margin-left: 20px;"><input type="radio" name="metric" value="emissions"> Emissions (tCO2e)</label>
        </div>

        <div id="facility-map"></div>

        <div class="status" id="data-status">Waiting for live data...</div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
    <script>
        console.log('🔧 Dashboard JavaScript loaded!');

        let map = null;
        let markerClusterGroup = null;
        let currentData = [];
        let currentMetric = 'power';

        // Wait for all libraries to load
        function waitForLibraries(callback) {
            if (typeof L !== 'undefined' && typeof L.markerClusterGroup === 'function') {
                console.log('📚 All libraries loaded (Leaflet + MarkerCluster)!');
                callback();
            } else {
                console.log('⏳ Waiting for libraries to load...');
                setTimeout(() => waitForLibraries(callback), 100);
            }
        }

        // Initialize map when DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            console.log('📄 DOM loaded, waiting for libraries...');
            waitForLibraries(() => {
                console.log('🚀 All libraries ready, initializing map...');
                initializeMap();
                startDataFetching();

                // Listen for metric changes
                document.querySelectorAll('input[name="metric"]').forEach(radio => {
                    radio.addEventListener('change', (e) => {
                        currentMetric = e.target.value;
                        console.log('🔄 Metric changed to:', currentMetric);
                        updateMapMarkers(currentData, currentMetric);
                    });
                });
            });
        });

        function initializeMap() {
            console.log('🏗️ Creating Leaflet map...');

            try {
                // Create map
                map = L.map('facility-map', {
                    center: [-25.2744, 133.7751],
                    zoom: 4,
                    zoomControl: true
                });

                // Add tile layer
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 19
                }).addTo(map);

                // Create marker cluster group
                markerClusterGroup = L.markerClusterGroup();
                map.addLayer(markerClusterGroup);

                console.log('✅ Map initialized successfully!');
            } catch (error) {
                console.error('❌ Error initializing map:', error);
            }
        }

        function startDataFetching() {
            console.log('🚀 Starting data fetch every 1 second...');
            fetchLiveData();
            setInterval(fetchLiveData, 1000);
        }

        function fetchLiveData() {
            console.log('🔄 Fetching data from API...');
            fetch('/api/live-data')
                .then(response => {
                    console.log('📡 API response status:', response.status);
                    return response.json();
                })
                .then(data => {
                    console.log('📊 Raw API response:', data);
                    currentData = data.data;
                    console.log('✅ Processed', currentData.length, 'facilities at', new Date().toLocaleTimeString());

                    // Log detailed data structure
                    if (currentData.length > 0) {
                        console.log('🔍 First facility structure:', currentData[0]);
                        const withCoords = currentData.filter(f => f.latitude !== null && f.longitude !== null);
                        console.log('📍 Facilities with coordinates:', withCoords.length, 'out of', currentData.length);
                    }

                    updateMapMarkers(currentData, currentMetric);
                    updateStatus(currentData);
                })
                .catch(error => {
                    console.error('❌ Fetch error:', error);
                    console.error('❌ Error details:', error.message);
                });
        }

        function updateMapMarkers(data, metric) {
            if (!map || !markerClusterGroup) {
                console.log('⚠️ Map not ready yet');
                return;
            }

            console.log('🎯 Updating markers for', data.length, 'facilities, metric:', metric);

            // Clear existing markers
            markerClusterGroup.clearLayers();

            // Filter valid coordinates
            const validData = data.filter(item =>
                item.latitude !== null && item.longitude !== null
            );

            console.log('📍 Valid facilities with coordinates:', validData.length);

            if (validData.length === 0) {
                console.log('⚠️ No facilities with coordinates - showing message on map');
                const noDataMarker = L.marker([-25.2744, 133.7751]).addTo(map)
                    .bindPopup('No facility locations available')
                    .openPopup();
                return;
            }

            // Create markers
            validData.forEach((item, index) => {
                if (index < 3) { // Log first 3 for debugging
                    console.log(`📍 Processing marker ${index + 1}:`, {
                        name: item.name,
                        lat: item.latitude,
                        lng: item.longitude,
                        power: item.power,
                        emissions: item.emissions
                    });
                }

                const value = item[metric] || 0;
                const color = value > 0 ? '#2ca02c' : '#d62728'; // Green for positive, red for negative/zero
                const radius = Math.max(3, Math.min(15, Math.abs(value) / 100)); // Scale size

                if (index < 3) {
                    console.log(`🎨 Marker ${index + 1} style:`, { color, radius, value });
                }

                const marker = L.circleMarker([item.latitude, item.longitude], {
                    radius: radius,
                    fillColor: color,
                    color: color,
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.7
                });

                // Add popup
                const popupContent = `
                    <b>${item.name || item.facility_id || 'Unknown'}</b><br>
                    Power: ${item.power || 'N/A'} MW<br>
                    Emissions: ${item.emissions || 'N/A'} tCO2e<br>
                    Region: ${item.network_region || 'N/A'}<br>
                    Coordinates: ${item.latitude}, ${item.longitude}
                `;
                marker.bindPopup(popupContent);

                markerClusterGroup.addLayer(marker);
            });

            console.log('✅ Successfully added', validData.length, 'markers to map');
        }

        function updateStatus(data) {
            const statusElement = document.getElementById('data-status');
            if (!statusElement) return;

            if (!data || data.length === 0) {
                statusElement.textContent = 'Waiting for live data...';
                return;
            }

            const validFacilities = data.filter(item =>
                item.latitude !== null && item.longitude !== null
            ).length;

            const timestamp = new Date().toLocaleTimeString();
            statusElement.textContent = `Last update: ${timestamp} · ${validFacilities} facilities with coordinates`;
            console.log('📊 Status updated:', statusElement.textContent);
        }
    </script>
</body>
</html>
        """

    # Set the layout to empty since we're using the custom route
    app.layout = html.Div()

    # Store references for API endpoint
    global_subscriber = subscriber
    global_metadata = facility_metadata

    # Add REST API endpoint for JavaScript to fetch data
    @app.server.route('/api/live-data')
    def get_live_data():
        """REST API endpoint that returns JSON data for JavaScript."""
        live_data = global_subscriber.store.snapshot()
        combined = _prepare_live_dataframe(live_data, global_metadata)

        # Convert DataFrame to JSON-serializable format
        data = []
        if not combined.empty:
            for _, row in combined.iterrows():
                # Handle NaN values properly for JSON serialization
                def safe_value(val):
                    if pd.isna(val):
                        return None
                    return val

                record = {
                    "facility_id": safe_value(row.get("facility_id")),
                    "name": safe_value(row.get("name")),
                    "fuel_type": safe_value(row.get("fuel_type")),
                    "network_region": safe_value(row.get("network_region")),
                    "latitude": float(row.get("latitude")) if pd.notna(row.get("latitude")) else None,
                    "longitude": float(row.get("longitude")) if pd.notna(row.get("longitude")) else None,
                    "power": float(row.get("power")) if pd.notna(row.get("power")) else None,
                    "emissions": float(row.get("emissions")) if pd.notna(row.get("emissions")) else None,
                    "timestamp": row.get("timestamp").isoformat() if pd.notna(row.get("timestamp")) else None,
                }
                data.append(record)

        return jsonify({"data": data})

    # No callbacks for now - just test basic layout


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
