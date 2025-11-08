from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

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
        ).add_to(map_obj)

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
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            min-height: 100vh;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #1a1a1a;
            margin: 0 0 20px 0;
            font-size: 1.8em;
            font-weight: 500;
            letter-spacing: -0.5px;
        }
        #facility-map {
            height: 75vh;
            min-height: 500px;
            border: 1px solid #e0e0e0;
            margin-top: 15px;
            position: relative;
        }

        /* Responsive design improvements */
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 20px;
            }
            #facility-map {
                height: 70vh;
                min-height: 400px;
            }
            .filters {
                flex-direction: column;
                gap: 15px;
            }
            .filter-group {
                width: 100%;
                min-width: unset;
            }
        }

        @media (max-width: 480px) {
            h1 {
                font-size: 1.8em;
            }
            .status {
                padding: 10px;
                font-size: 14px;
            }
        }
        .status {
            text-align: center;
            margin: 15px 0;
            padding: 10px;
            background: #fafafa;
            border-top: 1px solid #e0e0e0;
            font-size: 0.9em;
            color: #666;
        }
        .filters {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .filter-group {
            display: flex;
            flex-direction: column;
            min-width: 200px;
            padding: 0;
        }
        .filter-group label {
            font-weight: 500;
            margin-bottom: 6px;
            color: #333;
            font-size: 0.9em;
        }
        .dropdown-container {
            position: relative;
        }
        .dropdown-button {
            width: 100%;
            padding: 8px 12px;
            background: white;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            text-align: left;
            cursor: pointer;
            font-size: 0.9em;
            color: #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: border-color 0.2s ease;
        }
        .dropdown-button:hover {
            border-color: #999;
        }
        .dropdown-button.active {
            border-color: #666;
        }
        .dropdown-arrow {
            transition: transform 0.2s ease;
        }
        .dropdown-arrow.active {
            transform: rotate(180deg);
        }
        .dropdown-content {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-height: 250px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
            margin-top: 2px;
        }
        .dropdown-content.show {
            display: block;
        }
        .checkbox-group {
            padding: 8px;
        }
        .checkbox-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            padding: 4px;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        .checkbox-item:hover {
            background-color: #f8f9fa;
        }
        .checkbox-item input[type="checkbox"] {
            margin-right: 10px;
            transform: scale(1.2);
            accent-color: #3498db;
        }
        .checkbox-item label {
            margin: 0;
            font-size: 14px;
            cursor: pointer;
            color: #495057;
            font-weight: 500;
        }
        .metric-controls {
            text-align: center;
            margin-bottom: 15px;
            padding: 10px;
        }
        .metric-controls label {
            margin: 0 10px;
            font-weight: 400;
            color: #666;
            cursor: pointer;
            transition: color 0.2s;
            font-size: 0.9em;
            padding: 6px 12px;
            border-radius: 4px;
        }
        .metric-controls label:hover {
            color: #333;
            background: #f5f5f5;
        }
        .metric-controls input[type="radio"] {
            margin-right: 5px;
        }
        .metric-controls input[type="radio"]:checked + span {
            font-weight: 500;
            color: #1a1a1a;
        }
        .legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 1000;
            font-size: 12px;
            max-width: 180px;
            border: 1px solid #e0e0e0;
        }
        .legend h4 {
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #1a1a1a;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 4px;
            font-weight: 500;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 4px;
            padding: 2px;
        }
        .legend-color {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            border: 1px solid rgba(255,255,255,0.9);
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }

        .summary-panel {
            background: white;
            padding: 15px 0;
            margin-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
        .summary-panel h3 {
            margin: 0 0 12px 0;
            color: #1a1a1a;
            font-size: 1.1em;
            font-weight: 500;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .summary-table th, .summary-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .summary-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        .summary-table .total-row {
            background: #e8f4f8;
            font-weight: 600;
            color: #2c3e50;
        }
        .summary-table .total-row td {
            border-top: 2px solid #3498db;
        }
        .summary-table .market-row {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .summary-table .market-row td {
            font-weight: bold;
            text-align: center;
            padding: 12px;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .filters {
                flex-direction: column;
                gap: 15px;
            }
            .filter-group {
                min-width: auto;
            }
            .container {
                padding: 20px;
            }
            h1 {
                font-size: 2em;
            }
        }

        /* Custom scrollbar */
        .checkbox-group::-webkit-scrollbar {
            width: 6px;
        }
        .checkbox-group::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        .checkbox-group::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        .checkbox-group::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }

        /* Minimalist marker styling */
        .facility-marker {
            cursor: pointer;
            transition: opacity 0.2s ease;
        }

        .facility-marker:hover {
            opacity: 1 !important;
            filter: brightness(1.15);
        }

        /* Clean cluster styling */
        .marker-cluster {
            transition: transform 0.2s ease;
        }

        .marker-cluster:hover {
            transform: scale(1.05);
        }

        /* Loading animation for better UX */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .loading-pulse {
            animation: pulse 1.5s ease-in-out infinite;
        }

        /* Enhanced popup styling */
        .leaflet-popup-content-wrapper {
            border-radius: 8px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }

        .leaflet-popup-content {
            font-family: 'Segoe UI', sans-serif;
            line-height: 1.4;
        }

        /* Loading animation for status updates */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .status {
            animation: fadeInUp 0.5s ease-out;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Electricity Facilities Dashboard</h1>

        <div class="metric-controls">
            <label><input type="radio" name="metric" value="power" checked><span>Power (MW)</span></label>
            <label><input type="radio" name="metric" value="emissions"><span>Emissions (tCO₂)</span></label>
            <label><input type="radio" name="metric" value="price"><span>Price ($/MWh)</span></label>
            <label><input type="radio" name="metric" value="demand"><span>Demand (GW)</span></label>
        </div>

        <div class="filters">
            <div class="filter-group">
                <label>Filter by Region:</label>
                <div class="dropdown-container">
                    <button id="region-dropdown-btn" class="dropdown-button">
                        <span id="region-selected-text">All Regions</span>
                        <span class="dropdown-arrow">▼</span>
                    </button>
                    <div id="region-filter" class="dropdown-content checkbox-group"></div>
                </div>
            </div>

            <div class="filter-group">
                <label>Filter by Fuel Type:</label>
                <div class="dropdown-container">
                    <button id="fuel-dropdown-btn" class="dropdown-button">
                        <span id="fuel-selected-text">All Fuel Types</span>
                        <span class="dropdown-arrow">▼</span>
                    </button>
                    <div id="fuel-filter" class="dropdown-content checkbox-group"></div>
                </div>
            </div>
        </div>

        <div id="facility-map" style="position: relative;">
            <div class="legend" id="map-legend">
                <h4>Fuel Types</h4>
                <div id="legend-content">
                    <!-- Legend items will be populated by JavaScript -->
                </div>
            </div>
        </div>

        <div class="summary-panel" id="summary-panel">
            <h3>Regional Summary</h3>
            <div id="summary-content">
                <!-- Summary data will be populated by JavaScript -->
            </div>
        </div>

        <div class="status" id="data-status">Waiting for live data...</div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css">
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css">
    <script>
        console.log('🔧 Dashboard JavaScript loaded!');

        let map = null;
        let markerClusterGroup = null;
        let currentData = [];
        let marketData = {};
        let currentMetric = 'power';
        let selectedRegions = []; // Start with no filters (show all)
        let selectedFuelTypes = []; // Start with no filters (show all)
        let lastFilteredData = null; // Track last rendered data to avoid unnecessary updates
        let lastMetric = null; // Track last metric used
        let updateTimeout = null; // For debounced updates
        let isUpdating = false; // Prevent concurrent updates
        let isHandlingFilterChange = false; // Prevent dropdown closing during filter changes

        // Clean, professional color mapping for fuel types
        const fuelColors = {
            'Coal': '#5D4037',      // Brown
            'Gas': '#FF6F00',       // Deep Orange
            'Wind': '#43A047',      // Green
            'Solar': '#FBC02D',     // Yellow
            'Hydro': '#1976D2',     // Blue
            'Nuclear': '#7B1FA2',   // Purple
            'Oil': '#F57C00',       // Orange
            'Biomass': '#795548',   // Brown
            'Other': '#757575',     // Gray
            'default': '#546E7A'    // Blue Gray
        };

        // Metric-specific colors for when viewing market data
        const metricColors = {
            'power': '#2E7D32',     // Green for power
            'emissions': '#D32F2F', // Red for emissions
            'price': '#F57C00',     // Orange for price
            'demand': '#7B1FA2'     // Purple for demand
        };

        // Helper function to compare arrays
        function arraysEqual(a, b) {
            if (a.length !== b.length) return false;
            for (let i = 0; i < a.length; i++) {
                if (a[i] !== b[i]) return false;
            }
            return true;
        }

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
                        updateSummaryPanel(currentData);
                    });
                });

                // Listen for dropdown button clicks
                document.getElementById('region-dropdown-btn').addEventListener('click', () => toggleDropdown('region'));
                document.getElementById('fuel-dropdown-btn').addEventListener('click', () => toggleDropdown('fuel'));

                // Close dropdowns when clicking outside
                document.addEventListener('click', (e) => {
                    const clickedElement = e.target;

                    // Don't close dropdowns if we're in the middle of handling a filter change
                    if (isHandlingFilterChange) {
                        return;
                    }

                    // Don't close dropdowns if clicking on:
                    // 1. Checkboxes or their labels
                    // 2. Inside open dropdown content
                    if (clickedElement.type === 'checkbox' ||
                        clickedElement.tagName === 'LABEL' ||
                        clickedElement.closest('.dropdown-content.show')) {
                        return; // Don't close dropdowns
                    }

                    // Close dropdowns if clicking outside dropdown containers
                    if (!clickedElement.closest('.dropdown-container')) {
                        closeAllDropdowns();
                    }
                });

                // Listen for filter changes
                document.getElementById('region-filter').addEventListener('change', handleFilterChange);
                document.getElementById('fuel-filter').addEventListener('change', handleFilterChange);
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

                // Create marker cluster group with enhanced styling
                markerClusterGroup = L.markerClusterGroup({
                    showCoverageOnHover: false,
                    spiderfyOnMaxZoom: true,
                    spiderfyDistanceMultiplier: 2,
                    disableClusteringAtZoom: 10, // Disperse earlier at zoom level 10
                    maxClusterRadius: 60, // Tighter clustering
                    animate: true,
                    animateAddingMarkers: true,
                    iconCreateFunction: function(cluster) {
                        const childCount = cluster.getChildCount();
                        let size = 'small';
                        let iconSize = 35;
                        
                        if (childCount >= 50) {
                            size = 'large';
                            iconSize = 50;
                        } else if (childCount >= 10) {
                            size = 'medium';
                            iconSize = 42;
                        }
                        
                        return L.divIcon({
                            html: '<div style="width:' + iconSize + 'px; height:' + iconSize + 'px; display:flex; align-items:center; justify-content:center; border-radius:50%; background:rgba(52,152,219,0.8); color:white; font-weight:600; font-size:' + (iconSize/3) + 'px; border:3px solid white; box-shadow:0 2px 8px rgba(0,0,0,0.3);">' + childCount + '</div>',
                            className: 'marker-cluster marker-cluster-' + size,
                            iconSize: L.point(iconSize, iconSize)
                        });
                    }
                });
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
                    marketData = data.market_data || {};
                    console.log('✅ Processed', currentData.length, 'facilities at', new Date().toLocaleTimeString());
                    console.log('💰 Market data:', marketData);

                    // Log detailed data structure
                    if (currentData.length > 0) {
                        console.log('🔍 First facility structure:', currentData[0]);
                        const withCoords = currentData.filter(f => f.latitude !== null && f.longitude !== null);
                        console.log('📍 Facilities with coordinates:', withCoords.length, 'out of', currentData.length);
                        
                        // Log emissions data
                        const withEmissions = currentData.filter(f => f.emissions !== null && f.emissions !== undefined);
                        const withNonZeroEmissions = currentData.filter(f => f.emissions !== null && f.emissions !== undefined && f.emissions !== 0);
                        console.log('🌡️ Facilities with emissions data:', withEmissions.length);
                        console.log('🌡️ Facilities with non-zero emissions:', withNonZeroEmissions.length);
                        if (withNonZeroEmissions.length > 0) {
                            console.log('🌡️ Sample non-zero emissions:', withNonZeroEmissions.slice(0, 5).map(f => ({
                                id: f.facility_id,
                                name: f.name,
                                emissions: f.emissions
                            })));
                        }
                    }

                    // Populate filters on first data load
                    populateFilters(currentData);

                    updateMapMarkers(currentData, currentMetric);
                    updateStatus(currentData);
                    updateSummaryPanel(currentData);
                })
                .catch(error => {
                    console.error('❌ Fetch error:', error);
                    console.error('❌ Error details:', error.message);
                });
        }

        // Dropdown toggle functions
        function toggleDropdown(type) {
            const dropdown = document.getElementById(`${type}-filter`);
            const button = document.getElementById(`${type}-dropdown-btn`);
            const arrow = button.querySelector('.dropdown-arrow');

            // Toggle current dropdown (don't close others - allow multiple open)
            if (dropdown.classList.contains('show')) {
                dropdown.classList.remove('show');
                button.classList.remove('active');
                arrow.classList.remove('active');
            } else {
                dropdown.classList.add('show');
                button.classList.add('active');
                arrow.classList.add('active');
            }
        }

        function closeAllDropdowns() {
            document.querySelectorAll('.dropdown-content').forEach(dropdown => {
                dropdown.classList.remove('show');
            });
            document.querySelectorAll('.dropdown-button').forEach(button => {
                button.classList.remove('active');
                button.querySelector('.dropdown-arrow').classList.remove('active');
            });
        }

        // Update dropdown button text based on selected items
        function updateDropdownText(type) {
            const selected = type === 'region' ? selectedRegions : selectedFuelTypes;
            const totalOptions = document.querySelectorAll(`#${type}-filter input[type="checkbox"]`).length;
            const textElement = document.getElementById(`${type}-selected-text`);

            if (selected.length === 0) {
                textElement.textContent = `All ${type === 'region' ? 'Regions' : 'Fuel Types'}`;
            } else if (selected.length === totalOptions) {
                textElement.textContent = `All ${type === 'region' ? 'Regions' : 'Fuel Types'}`;
            } else if (selected.length === 1) {
                textElement.textContent = selected[0];
            } else {
                textElement.textContent = `${selected.length} selected`;
            }
        }

        // Populate filter checkboxes with available options
        function populateFilters(data) {
            // Store current selections before repopulating
            const currentRegionSelections = [...selectedRegions];
            const currentFuelSelections = [...selectedFuelTypes];

            // Get unique regions from actual data
            const regions = [...new Set(data.map(item => item.network_region).filter(r => r !== null && r !== undefined))].sort();
            const regionContainer = document.getElementById('region-filter');

            // Clear existing checkboxes
            regionContainer.innerHTML = '';

            // Add region checkboxes
            regions.forEach(region => {
                const checkboxItem = document.createElement('div');
                checkboxItem.className = 'checkbox-item';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `region-${region}`;
                checkbox.value = region;
                // Restore previous selection if this region was selected
                if (currentRegionSelections.includes(region)) {
                    checkbox.checked = true;
                }
                checkbox.addEventListener('change', handleFilterChange);

                const label = document.createElement('label');
                label.htmlFor = `region-${region}`;
                label.textContent = region;

                checkboxItem.appendChild(checkbox);
                checkboxItem.appendChild(label);
                regionContainer.appendChild(checkboxItem);
            });

            // Get unique fuel types from actual data
            const fuelTypes = [...new Set(data.map(item => item.fuel_type).filter(f => f !== null && f !== undefined))].sort();
            const fuelContainer = document.getElementById('fuel-filter');

            // Clear existing checkboxes
            fuelContainer.innerHTML = '';

            // Add fuel type checkboxes
            fuelTypes.forEach(fuel => {
                const checkboxItem = document.createElement('div');
                checkboxItem.className = 'checkbox-item';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `fuel-${fuel}`;
                checkbox.value = fuel;
                // Restore previous selection if this fuel type was selected
                if (currentFuelSelections.includes(fuel)) {
                    checkbox.checked = true;
                }
                checkbox.addEventListener('change', handleFilterChange);

                const label = document.createElement('label');
                label.htmlFor = `fuel-${fuel}`;
                label.textContent = fuel;

                checkboxItem.appendChild(checkbox);
                checkboxItem.appendChild(label);
                fuelContainer.appendChild(checkboxItem);
            });

            // Update the global selected arrays with restored values
            selectedRegions = currentRegionSelections.filter(region => regions.includes(region));
            selectedFuelTypes = currentFuelSelections.filter(fuel => fuelTypes.includes(fuel));

            console.log('🔧 Filter checkboxes populated from data:', {
                regions: regions,
                fuelTypes: fuelTypes,
                regionCount: regions.length,
                fuelTypeCount: fuelTypes.length,
                restoredRegionSelections: selectedRegions,
                restoredFuelSelections: selectedFuelTypes
            });

            // Update dropdown texts with restored selections
            updateDropdownText('region');
            updateDropdownText('fuel');
        }

        // Apply filters to data
        function applyFilters(data) {
            let filtered = data;

            // Apply region filter only if regions are selected
            if (selectedRegions.length > 0) {
                filtered = filtered.filter(item =>
                    selectedRegions.includes(item.network_region)
                );
            }

            // Apply fuel type filter only if fuel types are selected
            if (selectedFuelTypes.length > 0) {
                filtered = filtered.filter(item =>
                    selectedFuelTypes.includes(item.fuel_type)
                );
            }

            return filtered;
        }

        // Handle filter changes
        function handleFilterChange(event) {
            isHandlingFilterChange = true;

            // Update selected regions from checkboxes
            selectedRegions = Array.from(document.querySelectorAll('#region-filter input[type="checkbox"]:checked'))
                .map(checkbox => checkbox.value);

            // Update selected fuel types from checkboxes
            selectedFuelTypes = Array.from(document.querySelectorAll('#fuel-filter input[type="checkbox"]:checked'))
                .map(checkbox => checkbox.value);

            // Update dropdown button texts
            updateDropdownText('region');
            updateDropdownText('fuel');

            console.log('🔄 Filters updated:', {
                regions: selectedRegions,
                fuelTypes: selectedFuelTypes,
                showAll: selectedRegions.length === 0 && selectedFuelTypes.length === 0
            });

            // Debounced update to prevent excessive re-rendering
            debouncedUpdate();

            // Reset flag after a short delay to allow UI updates
            setTimeout(() => {
                isHandlingFilterChange = false;
            }, 100);
        }

        // Debounced update function for performance
        function debouncedUpdate() {
            if (updateTimeout) {
                clearTimeout(updateTimeout);
            }
            updateTimeout = setTimeout(() => {
                updateMapMarkers(currentData, currentMetric);
                updateSummaryPanel(currentData);
            }, 300); // 300ms debounce
        }

        function updateMapMarkers(data, metric) {
            if (!map || !markerClusterGroup) {
                console.log('⚠️ Map not ready yet');
                return;
            }

            // Prevent concurrent updates for better performance
            if (isUpdating) {
                console.log('⏳ Update already in progress, skipping...');
                return;
            }

            isUpdating = true;
            const startTime = performance.now();

            // Show loading indicator
            showLoadingIndicator(true);

            // Apply filters first
            const filteredData = applyFilters(data);

            // Check if data or metric has changed to avoid unnecessary re-renders
            const dataChanged = !lastFilteredData ||
                lastFilteredData.length !== filteredData.length ||
                lastMetric !== metric ||
                !arraysEqual(lastFilteredData.map(d => d.facility_id), filteredData.map(d => d.facility_id));

            if (!dataChanged) {
                console.log('🔄 Data unchanged, skipping marker update');
                return;
            }

            console.log('🎯 Updating markers for', filteredData.length, 'filtered facilities, metric:', metric);

            // Store current state for next comparison
            lastFilteredData = filteredData.slice(); // Clone array
            lastMetric = metric;

            // Clear existing markers
            markerClusterGroup.clearLayers();

            // Filter valid coordinates
            const validData = filteredData.filter(item =>
                item.latitude !== null && item.longitude !== null
            );

            console.log('📍 Valid facilities with coordinates after filtering:', validData.length);

            // Calculate density for each marker (number of nearby facilities)
            const calculateDensity = (facility, allFacilities, radiusKm = 10) => {
                let count = 0;
                allFacilities.forEach(other => {
                    if (other.facility_id !== facility.facility_id &&
                        other.latitude !== null && other.longitude !== null) {
                        const distance = getDistance(
                            facility.latitude, facility.longitude,
                            other.latitude, other.longitude
                        );
                        if (distance <= radiusKm) {
                            count++;
                        }
                    }
                });
                return count;
            };

            // Haversine distance calculation
            const getDistance = (lat1, lon1, lat2, lon2) => {
                const R = 6371; // Earth's radius in km
                const dLat = (lat2 - lat1) * Math.PI / 180;
                const dLon = (lon2 - lon1) * Math.PI / 180;
                const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                    Math.sin(dLon/2) * Math.sin(dLon/2);
                const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
                return R * c;
            };

            if (validData.length === 0) {
                console.log('⚠️ No facilities match current filters');
                const noDataMarker = L.marker([-25.2744, 133.7751]).addTo(map)
                    .bindPopup('No facilities match current filters')
                    .openPopup();
                return;
            }

            // Calculate size scaling based on metric values
            let values, maxValue;
            if (metric === 'price' || metric === 'demand') {
                // Market-wide metrics - all facilities have the same value
                const marketValue = marketData[metric] || 0;
                values = [Math.abs(marketValue)];
                maxValue = Math.abs(marketValue) || 1;
            } else {
                // Facility-specific metrics
                values = validData.map(item => Math.abs(item[metric] || 0));
                maxValue = Math.max(...values) || 1;
            }
            const minSize = 6;
            const maxSize = 14;

            // Create markers with density-based enhancements
            validData.forEach((item, index) => {
                // Calculate density for this facility (performance optimization: sample every 10th for large datasets)
                const densitySample = validData.length > 100 ? validData.filter((_, i) => i % 10 === 0) : validData;
                const density = calculateDensity(item, densitySample, 15); // 15km radius for density calculation

                if (index < 3) { // Log first 3 for debugging
                    console.log(`📍 Processing marker ${index + 1}:`, {
                        name: item.name,
                        fuel: item.fuel_type,
                        region: item.network_region,
                        lat: item.latitude,
                        lng: item.longitude,
                        power: item.power,
                        emissions: item.emissions,
                        density: density
                    });
                }

                // Color based on metric type for market data, fuel type for facility data
                const color = (metric === 'price' || metric === 'demand')
                    ? metricColors[metric]
                    : (fuelColors[item.fuel_type] || fuelColors.default);

                // Enhanced size calculation: combine metric value with density factor
                let baseRadius, value;
                if (metric === 'price' || metric === 'demand') {
                    // Market-wide metrics - base size with density enhancement
                    const marketValue = marketData[metric] || 0;
                    value = marketValue;
                    if (marketValue === 0) {
                        baseRadius = minSize;
                    } else {
                        // Scale market value for visualization
                        const scaledValue = metric === 'price' ? marketValue : marketValue / 1000; // Convert demand to GW
                        const logValue = Math.log10(Math.abs(scaledValue) + 1);
                        const maxLogValue = Math.log10(Math.abs(maxValue) + 1);
                        baseRadius = minSize + (logValue / maxLogValue) * (maxSize - minSize);
                    }
                } else {
                    // Facility-specific metrics with density enhancement
                    value = Math.abs(item[metric] || 0);
                    if (value === 0) {
                        baseRadius = minSize;
                    } else {
                        // Use logarithmic scaling to better show differences
                        const logValue = Math.log10(value + 1);
                        const maxLogValue = Math.log10(maxValue + 1);
                        baseRadius = minSize + (logValue / maxLogValue) * (maxSize - minSize);
                    }
                }

                // Apply density factor: facilities in denser areas get slightly larger markers
                const densityFactor = 1 + Math.min(density * 0.1, 0.5); // Max 50% increase for very dense areas
                let radius = baseRadius * densityFactor;
                radius = Math.max(minSize, Math.min(maxSize * 1.5, radius)); // Allow larger max for dense areas

                if (index < 3) {
                    console.log(`🎨 Marker ${index + 1} style:`, {
                        fuel: item.fuel_type,
                        color,
                        value,
                        radius: radius.toFixed(1)
                    });
                }

                const marker = L.circleMarker([item.latitude, item.longitude], {
                    radius: radius,
                    fillColor: color,
                    color: '#ffffff',
                    weight: 1.5,
                    opacity: 0.9,
                    fillOpacity: 0.7,
                    className: 'facility-marker'
                });

                // Add detailed popup with all facility information
                const facilityName = item.name || item.facility_id || 'Unknown';
                const fuelType = item.fuel_type || 'Unknown';
                const currentPower = item.power !== null ? item.power.toFixed(2) + ' MW' : 'N/A';

                // Debug emissions value
                let emissions = 'N/A';
                if (item.emissions !== null && item.emissions !== undefined) {
                    emissions = item.emissions.toFixed(2) + ' tonnes';
                    if (index < 5) {
                        console.log(`🔍 Popup ${index + 1}: ${facilityName} - emissions=${item.emissions}, formatted=${emissions}`);
                    }
                } else {
                    if (index < 5) {
                        console.log(`⚠️ Popup ${index + 1}: ${facilityName} - emissions is null/undefined`);
                    }
                }
                const region = item.network_region || 'N/A';
                const lastUpdate = item.timestamp ? new Date(item.timestamp).toLocaleString('en-US', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: true
                }) : 'N/A';

                const fuelColor = fuelColors[item.fuel_type] || fuelColors.default;
                const popupContent = `
                    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 280px; line-height: 1.5;">
                        <div style="background: #f5f5f5; padding: 10px; margin-bottom: 10px; border-left: 3px solid ${fuelColor};">
                            <div style="font-size: 15px; font-weight: 500; color: #1a1a1a;">${facilityName}</div>
                            <div style="font-size: 12px; color: #666; margin-top: 2px;">${fuelType}</div>
                        </div>
                        <div style="padding: 0 4px;">
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 8px; font-size: 13px;">
                                <span style="color: #666;">Power:</span>
                                <span style="font-weight: 500; color: #2E7D32; text-align: right;">${currentPower}</span>
                                
                                <span style="color: #666;">Emissions:</span>
                                <span style="font-weight: 500; color: #D32F2F; text-align: right;">${emissions}</span>
                                
                                <span style="color: #666;">Region:</span>
                                <span style="text-align: right;">${region}</span>
                            </div>
                            <div style="border-top: 1px solid #e0e0e0; margin: 10px 0;"></div>
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 8px; font-size: 13px;">
                                <span style="color: #666;">Market Price:</span>
                                <span style="font-weight: 500; color: #F57C00; text-align: right;">$${marketData.price ? marketData.price.toFixed(2) : 'N/A'}/MWh</span>
                                
                                <span style="color: #666;">Demand:</span>
                                <span style="font-weight: 500; color: #7B1FA2; text-align: right;">${marketData.demand ? (marketData.demand / 1000).toFixed(1) : 'N/A'} GW</span>
                            </div>
                            <div style="margin-top: 10px; font-size: 11px; color: #999; text-align: center;">
                                Updated: ${lastUpdate}
                            </div>
                        </div>
                    </div>
                `;
                // Bind popup with options to keep it open until manually closed
                marker.bindPopup(popupContent, {
                    closeButton: true,
                    autoClose: false,
                    closeOnEscapeKey: true,
                    closeOnClick: false
                });

                // Add marker to cluster group
                markerClusterGroup.addLayer(marker);
            });

            console.log('✅ Successfully added', validData.length, 'markers to map');

            // Update legend based on current metric
            updateLegend(validData, metric);

            // Hide loading indicator and log performance
            const endTime = performance.now();
            console.log(`⚡ Map update completed in ${(endTime - startTime).toFixed(2)}ms`);
            showLoadingIndicator(false);
            isUpdating = false;
        }

        // Loading indicator functions
        function showLoadingIndicator(show) {
            let indicator = document.getElementById('map-loading-indicator');
            if (!indicator) {
                indicator = document.createElement('div');
                indicator.id = 'map-loading-indicator';
                indicator.style.cssText = `
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: rgba(255, 255, 255, 0.95);
                    padding: 8px 12px;
                    border-radius: 6px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    font-size: 12px;
                    color: #666;
                    z-index: 1000;
                    display: none;
                    font-family: 'Segoe UI', sans-serif;
                `;
                indicator.innerHTML = '🔄 Updating map...';
                document.getElementById('facility-map').style.position = 'relative';
                document.getElementById('facility-map').appendChild(indicator);
            }
            indicator.style.display = show ? 'block' : 'none';
        }

        // Update the summary panel with regional data
        function updateSummaryPanel(data) {
            const filteredData = applyFilters(data);
            const summaryContent = document.getElementById('summary-content');
            if (!summaryContent) return;

            // Group data by region
            const regionalData = {};
            filteredData.forEach(facility => {
                const region = facility.network_region || 'Unknown';
                if (!regionalData[region]) {
                    regionalData[region] = {
                        facilities: 0,
                        totalPower: 0,
                        totalEmissions: 0
                    };
                }
                regionalData[region].facilities += 1;
                if (facility.power !== null) {
                    regionalData[region].totalPower += facility.power;
                }
                if (facility.emissions !== null) {
                    regionalData[region].totalEmissions += facility.emissions;
                }
            });

            // Create summary table
            let tableHTML = `
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Region</th>
                            <th>Facilities</th>
                            <th>Total Power (MW)</th>
                            <th>Total CO₂ (tonnes)</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            // Add market data row if available
            if (marketData.price !== null && marketData.price !== undefined) {
                tableHTML += `
                    <tr class="market-row">
                        <td><strong>MARKET DATA</strong></td>
                        <td colspan="3" style="text-align: left; padding-left: 20px;">
                            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                                <span><strong>Price:</strong> $${marketData.price.toFixed(2)}/MWh</span>
                                <span><strong>Demand:</strong> ${(marketData.demand / 1000).toFixed(1)} GW</span>
                                <span><strong>Last Update:</strong> ${marketData.timestamp ? new Date(marketData.timestamp).toLocaleTimeString('en-US', {hour12: false}) : 'N/A'}</span>
                            </div>
                        </td>
                    </tr>
                `;
            }

            // Sort regions for consistent display
            const sortedRegions = Object.keys(regionalData).sort();

            sortedRegions.forEach(region => {
                const data = regionalData[region];
                tableHTML += `
                    <tr>
                        <td>${region}</td>
                        <td>${data.facilities}</td>
                        <td>${data.totalPower.toFixed(1)}</td>
                        <td>${data.totalEmissions.toFixed(1)}</td>
                    </tr>
                `;
            });

            // Add totals row
            const totals = sortedRegions.reduce((acc, region) => {
                const data = regionalData[region];
                acc.facilities += data.facilities;
                acc.totalPower += data.totalPower;
                acc.totalEmissions += data.totalEmissions;
                return acc;
            }, { facilities: 0, totalPower: 0, totalEmissions: 0 });

            tableHTML += `
                    <tr class="total-row">
                        <td><strong>TOTAL</strong></td>
                        <td><strong>${totals.facilities}</strong></td>
                        <td><strong>${totals.totalPower.toFixed(1)}</strong></td>
                        <td><strong>${totals.totalEmissions.toFixed(1)}</strong></td>
                    </tr>
                </tbody>
                </table>
            `;

            summaryContent.innerHTML = tableHTML;
            console.log('📊 Summary panel updated');
        }

        // Update the legend based on current metric
        function updateLegend(data, metric) {
            const legendContent = document.getElementById('legend-content');
            const legendTitle = document.querySelector('.legend h4');
            if (!legendContent || !legendTitle) return;

            legendContent.innerHTML = '';

            if (metric === 'price' || metric === 'demand') {
                // Show metric-specific legend for market data
                legendTitle.textContent = `${metric.charAt(0).toUpperCase() + metric.slice(1)} Visualization`;
                const color = metricColors[metric];
                const value = marketData[metric];
                const unit = metric === 'price' ? '$/MWh' : 'GW';
                const displayValue = value ? (metric === 'demand' ? (value / 1000).toFixed(1) : value.toFixed(2)) : 'N/A';

                const legendItem = document.createElement('div');
                legendItem.className = 'legend-item';
                legendItem.innerHTML = `
                    <div class="legend-color" style="background-color: ${color};"></div>
                    <span>Market ${metric.charAt(0).toUpperCase() + metric.slice(1)}: ${displayValue} ${unit}</span>
                `;
                legendContent.appendChild(legendItem);

                // Add note about uniform sizing
                const noteItem = document.createElement('div');
                noteItem.className = 'legend-item';
                noteItem.style.fontSize = '11px';
                noteItem.style.color = '#666';
                noteItem.style.marginTop = '8px';
                noteItem.innerHTML = '<span>All facilities show market-wide value</span>';
                legendContent.appendChild(noteItem);
            } else {
                // Show fuel type legend for facility data
                legendTitle.textContent = 'Fuel Types';
                const fuelTypes = [...new Set(data.map(item => item.fuel_type).filter(f => f !== null))].sort();

                if (fuelTypes.length === 0) {
                    legendContent.innerHTML = '<div style="color: #999; font-style: italic;">No data</div>';
                    return;
                }

                fuelTypes.forEach(fuelType => {
                    const color = fuelColors[fuelType] || fuelColors.default;
                    const legendItem = document.createElement('div');
                    legendItem.className = 'legend-item';
                    legendItem.innerHTML = `
                        <div class="legend-color" style="background-color: ${color};"></div>
                        <span>${fuelType}</span>
                    `;
                    legendContent.appendChild(legendItem);
                });
            }
        }

        function updateStatus(data) {
            const statusElement = document.getElementById('data-status');
            if (!statusElement) return;

            if (!data || data.length === 0) {
                statusElement.textContent = 'Waiting for live data...';
                return;
            }

            const filteredData = applyFilters(data);
            const validFacilities = filteredData.filter(item =>
                item.latitude !== null && item.longitude !== null
            ).length;

            const totalFacilities = data.length;
            const totalWithCoords = data.filter(item =>
                item.latitude !== null && item.longitude !== null
            ).length;

            const timestamp = new Date().toLocaleTimeString();

            let statusHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div style="font-weight: bold; color: #2c3e50;">
                        Total Facilities: ${totalFacilities} |
                        With Coordinates: ${totalWithCoords}`;

            if (selectedRegions.length > 0 || selectedFuelTypes.length > 0) {
                statusHTML += ` | Filtered: ${validFacilities} shown`;
            } else {
                statusHTML += ` | All: ${validFacilities} shown`;
            }

            statusHTML += ` | Last Update: ${timestamp}`;

            // Add market data if available
            if (marketData.price !== null && marketData.price !== undefined) {
                statusHTML += `<br>Market Price: $${marketData.price.toFixed(2)}/MWh`;
            }
            if (marketData.demand !== null && marketData.demand !== undefined) {
                statusHTML += ` | Demand: ${(marketData.demand / 1000).toFixed(1)} GW`;
            }

            statusHTML += `</div></div>`;

            statusElement.innerHTML = statusHTML;
            console.log('📊 Status updated with market data:', marketData);
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

        # Always try to load cached data to supplement live data with emissions
        cached_data = None
        try:
            cache_dir = Path("cache")
            csv_files = list(cache_dir.glob("*_consolidated.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"Loading cached data from {latest_csv} to supplement emissions")
                cached_df = pd.read_csv(latest_csv)
                cached_df["timestamp"] = pd.to_datetime(cached_df["timestamp"])
                cached_data = cached_df
        except Exception as e:
            logger.warning(f"Could not load cached data: {e}")

        # Prepare live data first
        combined = _prepare_live_dataframe(live_data, global_metadata)

        # If we have cached data, supplement with cached facilities and emissions
        if cached_data is not None:
            # Prepare cached data with metadata
            cached_combined = _prepare_live_dataframe(cached_data, global_metadata)
            
            if not cached_combined.empty:
                # Get latest data per facility from cached (prioritize non-zero emissions)
                # Sort by facility_id, then by whether emissions is non-zero (True first), then by timestamp
                cached_combined["has_emissions"] = cached_combined["emissions"].notna() & (cached_combined["emissions"] != 0)
                cached_latest = cached_combined.sort_values(
                    ["facility_id", "has_emissions", "timestamp"],
                    ascending=[True, False, False]
                ).drop_duplicates(subset=["facility_id"], keep="first")
                cached_latest = cached_latest.drop(columns=["has_emissions"])
                
                # If we have live data, merge cached data to fill gaps
                if not combined.empty:
                    # Merge cached emissions/power where live data is missing or zero
                    cached_for_merge = cached_latest[["facility_id", "emissions", "power", "timestamp"]].copy()
                    combined = combined.merge(
                        cached_for_merge,
                        on="facility_id",
                        how="left",
                        suffixes=("", "_cached")
                    )
                    
                    # Fill missing or zero emissions with cached values (but don't overwrite non-zero live emissions)
                    missing_emissions = (combined["emissions"].isna() | (combined["emissions"] == 0)) & combined["emissions_cached"].notna() & (combined["emissions_cached"] != 0)
                    if missing_emissions.any():
                        combined.loc[missing_emissions, "emissions"] = combined.loc[missing_emissions, "emissions_cached"]
                        logger.info(f"Filled {missing_emissions.sum()} facilities with cached emissions data")
                    
                    # Also fill missing power if available
                    missing_power = combined["power"].isna() & combined["power_cached"].notna()
                    if missing_power.any():
                        combined.loc[missing_power, "power"] = combined.loc[missing_power, "power_cached"]
                    
                    # Update timestamp if cached is more recent
                    if "timestamp_cached" in combined.columns:
                        combined["timestamp"] = combined[["timestamp", "timestamp_cached"]].max(axis=1)
                    
                    # Clean up temporary columns
                    combined = combined.drop(columns=[col for col in combined.columns if col.endswith("_cached")])
                    
                    # Add facilities from cached data that aren't in live data
                    live_facility_ids = set(combined["facility_id"].unique())
                    cached_only = cached_latest[~cached_latest["facility_id"].isin(live_facility_ids)]
                    if not cached_only.empty:
                        logger.info(f"Adding {len(cached_only)} facilities from cached data not in live stream")
                        combined = pd.concat([combined, cached_only], ignore_index=True)
                else:
                    # No live data, use cached data entirely
                    logger.info("No live data, using cached data")
                    combined = cached_latest

        # Log data structure
        if not combined.empty:
            logger.info(f"Final combined data: {len(combined)} facilities")
            if "emissions" in combined.columns:
                non_zero_emissions = combined[combined["emissions"].notna() & (combined["emissions"] != 0)]
                logger.info(f"Facilities with non-zero emissions: {len(non_zero_emissions)}")
                if len(non_zero_emissions) > 0:
                    logger.info(f"Sample emissions: {non_zero_emissions['emissions'].head().tolist()}")
            with_coords = combined[combined["latitude"].notna() & combined["longitude"].notna()]
            logger.info(f"Facilities with coordinates: {len(with_coords)} out of {len(combined)}")
            
            # Log facilities without coordinates for debugging
            without_coords = combined[combined["latitude"].isna() | combined["longitude"].isna()]
            if len(without_coords) > 0:
                logger.info(f"Facilities without coordinates: {len(without_coords)}")
                logger.info(f"Sample facility IDs without coords: {without_coords['facility_id'].head(10).tolist()}")

        # Get latest market price and demand data
        latest_price = None
        latest_demand = None
        if not combined.empty and 'price' in combined.columns and 'demand' in combined.columns:
            # Calculate average price and demand across all facilities for the most recent timestamp
            if not combined.empty:
                # Sort by timestamp to get most recent data
                combined_sorted = combined.sort_values('timestamp', ascending=False)
                most_recent_timestamp = combined_sorted['timestamp'].iloc[0]

                # Get data for the most recent timestamp
                recent_data = combined_sorted[combined_sorted['timestamp'] == most_recent_timestamp]

                # Calculate market price and demand - ensure positive values
                price_data = recent_data['price'].dropna()
                demand_data = recent_data['demand'].dropna()

                if not price_data.empty:
                    # For market price, use absolute value to prevent negative prices
                    # Negative prices in electricity markets are unusual but can occur
                    # For display purposes, we'll use the absolute value
                    latest_price = float(abs(price_data.mean()))

                if not demand_data.empty:
                    # For demand, ensure non-negative values
                    valid_demand = demand_data[demand_data >= 0]
                    if not valid_demand.empty:
                        latest_demand = float(valid_demand.mean())
                    else:
                        latest_demand = float(demand_data.abs().mean()) if not demand_data.empty else None

        # Convert DataFrame to JSON-serializable format
        data = []
        if not combined.empty:
            # Mock fuel types and regions for demo purposes
            fuel_types = ['Coal', 'Gas', 'Wind', 'Solar', 'Hydro', 'Nuclear', 'Oil', 'Biomass']
            regions = ['NSW1', 'VIC1', 'QLD1', 'SA1', 'WA1', 'TAS1']

            for i, (_, row) in enumerate(combined.iterrows()):
                # Handle NaN values properly for JSON serialization
                def safe_value(val):
                    if pd.isna(val):
                        return None
                    return val

                # Add mock data for demo
                fuel_type = safe_value(row.get("fuel_type"))
                network_region = safe_value(row.get("network_region"))

                # If no fuel type, assign one based on facility ID hash
                if fuel_type is None:
                    fuel_type = fuel_types[hash(row.get("facility_id", "")) % len(fuel_types)]

                # If no region, assign one based on facility ID hash
                if network_region is None:
                    network_region = regions[hash(row.get("facility_id", "")) % len(regions)]

                # Get emissions value with proper handling
                emissions_value = row.get("emissions")
                if pd.notna(emissions_value):
                    emissions_val = float(emissions_value)
                else:
                    emissions_val = None

                record = {
                    "facility_id": safe_value(row.get("facility_id")),
                    "name": safe_value(row.get("name")),
                    "fuel_type": fuel_type,
                    "network_region": network_region,
                    "latitude": float(row.get("latitude")) if pd.notna(row.get("latitude")) else None,
                    "longitude": float(row.get("longitude")) if pd.notna(row.get("longitude")) else None,
                    "power": float(row.get("power")) if pd.notna(row.get("power")) else None,
                    "emissions": emissions_val,
                    "timestamp": row.get("timestamp").isoformat() if pd.notna(row.get("timestamp")) else None,
                }
                data.append(record)


        response_data = {
            "data": data,
            "market_data": {
                "price": latest_price,
                "demand": latest_demand,
                "timestamp": datetime.now().isoformat() if latest_price or latest_demand else None
            }
        }

        return jsonify(response_data)

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
