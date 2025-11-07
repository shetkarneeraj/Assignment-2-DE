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
        .filters { display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .filter-group { display: flex; flex-direction: column; min-width: 200px; }
        .filter-group label { font-weight: bold; margin-bottom: 5px; color: #34495e; }
        .filter-group select { padding: 8px; border: 1px solid #ddd; border-radius: 4px; background: white; min-height: 120px; }
        .filter-group select[multiple] { height: 120px; }
        .metric-controls { text-align: center; margin-bottom: 10px; }
        .metric-controls label { margin: 0 10px; }
        .legend { position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); z-index: 1000; font-size: 12px; max-width: 200px; }
        .legend h4 { margin: 0 0 8px 0; font-size: 14px; }
        .legend-item { display: flex; align-items: center; margin-bottom: 4px; }
        .legend-color { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Electricity Facilities Dashboard</h1>

        <div class="metric-controls">
            <label><input type="radio" name="metric" value="power" checked> Power (MW)</label>
            <label style="margin-left: 20px;"><input type="radio" name="metric" value="emissions"> Emissions (tCO2e)</label>
        </div>

        <div class="filters">
            <div class="filter-group">
                <label for="region-filter">Filter by Region (Multi-select):</label>
                <select id="region-filter" multiple>
                    <option value="all">All Regions</option>
                </select>
            </div>

            <div class="filter-group">
                <label for="fuel-filter">Filter by Fuel Type (Multi-select):</label>
                <select id="fuel-filter" multiple>
                    <option value="all">All Fuel Types</option>
                </select>
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
        let selectedRegions = ['all'];
        let selectedFuelTypes = ['all'];

        // Color mapping for fuel types (based on actual data)
        const fuelColors = {
            'Coal': '#8B4513',      // Brown
            'Gas': '#FF6B35',       // Orange-red
            'Wind': '#4CAF50',      // Green
            'Solar': '#FFD700',     // Gold
            'Hydro': '#2196F3',     // Blue
            'Nuclear': '#9C27B0',   // Purple
            'Oil': '#FF9800',       // Orange
            'Biomass': '#795548',   // Brown-gray
            'Other': '#9E9E9E',     // Gray
            'default': '#607D8B'    // Blue-gray
        };

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

                    // Populate filters on first data load
                    populateFilters(currentData);

                    updateMapMarkers(currentData, currentMetric);
                    updateStatus(currentData);
                })
                .catch(error => {
                    console.error('❌ Fetch error:', error);
                    console.error('❌ Error details:', error.message);
                });
        }

        // Populate filter dropdowns with available options
        function populateFilters(data) {
            // Get unique regions from actual data
            const regions = [...new Set(data.map(item => item.network_region).filter(r => r !== null && r !== undefined))].sort();
            const regionSelect = document.getElementById('region-filter');

            // Clear existing options except "All Regions"
            while (regionSelect.options.length > 1) {
                regionSelect.remove(1);
            }

            // Add region options
            regions.forEach(region => {
                const option = document.createElement('option');
                option.value = region;
                option.textContent = region;
                regionSelect.appendChild(option);
            });

            // Get unique fuel types from actual data
            const fuelTypes = [...new Set(data.map(item => item.fuel_type).filter(f => f !== null && f !== undefined))].sort();
            const fuelSelect = document.getElementById('fuel-filter');

            // Clear existing options except "All Fuel Types"
            while (fuelSelect.options.length > 1) {
                fuelSelect.remove(1);
            }

            // Add fuel type options
            fuelTypes.forEach(fuel => {
                const option = document.createElement('option');
                option.value = fuel;
                option.textContent = fuel;
                fuelSelect.appendChild(option);
            });

            console.log('🔧 Filters populated from data:', {
                regions: regions,
                fuelTypes: fuelTypes,
                regionCount: regions.length,
                fuelTypeCount: fuelTypes.length
            });
        }

        // Apply filters to data
        function applyFilters(data) {
            let filtered = data;

            // Apply region filter
            if (!selectedRegions.includes('all')) {
                filtered = filtered.filter(item =>
                    selectedRegions.includes(item.network_region)
                );
            }

            // Apply fuel type filter
            if (!selectedFuelTypes.includes('all')) {
                filtered = filtered.filter(item =>
                    selectedFuelTypes.includes(item.fuel_type)
                );
            }

            return filtered;
        }

        // Handle filter changes
        function handleFilterChange() {
            // Update selected regions
            const regionSelect = document.getElementById('region-filter');
            selectedRegions = Array.from(regionSelect.selectedOptions).map(option => option.value);
            if (selectedRegions.length === 0) selectedRegions = ['all'];

            // Update selected fuel types
            const fuelSelect = document.getElementById('fuel-filter');
            selectedFuelTypes = Array.from(fuelSelect.selectedOptions).map(option => option.value);
            if (selectedFuelTypes.length === 0) selectedFuelTypes = ['all'];

            console.log('🔄 Filters updated:', { regions: selectedRegions, fuelTypes: selectedFuelTypes });

            // Re-render markers with filters applied
            updateMapMarkers(currentData, currentMetric);
        }

        function updateMapMarkers(data, metric) {
            if (!map || !markerClusterGroup) {
                console.log('⚠️ Map not ready yet');
                return;
            }

            // Apply filters first
            const filteredData = applyFilters(data);
            console.log('🎯 Updating markers for', filteredData.length, 'filtered facilities, metric:', metric);

            // Clear existing markers
            markerClusterGroup.clearLayers();

            // Filter valid coordinates
            const validData = filteredData.filter(item =>
                item.latitude !== null && item.longitude !== null
            );

            console.log('📍 Valid facilities with coordinates after filtering:', validData.length);

            if (validData.length === 0) {
                console.log('⚠️ No facilities match current filters');
                const noDataMarker = L.marker([-25.2744, 133.7751]).addTo(map)
                    .bindPopup('No facilities match current filters')
                    .openPopup();
                return;
            }

            // Calculate size scaling based on metric values
            const values = validData.map(item => Math.abs(item[metric] || 0));
            const maxValue = Math.max(...values) || 1;
            const minSize = 4;
            const maxSize = 20;

            // Create markers
            validData.forEach((item, index) => {
                if (index < 3) { // Log first 3 for debugging
                    console.log(`📍 Processing marker ${index + 1}:`, {
                        name: item.name,
                        fuel: item.fuel_type,
                        region: item.network_region,
                        lat: item.latitude,
                        lng: item.longitude,
                        power: item.power,
                        emissions: item.emissions
                    });
                }

                const value = Math.abs(item[metric] || 0);
                const color = fuelColors[item.fuel_type] || fuelColors.default;

                // Size proportional to output (logarithmic scaling for better visualization)
                let radius;
                if (value === 0) {
                    radius = minSize;
                } else {
                    // Use logarithmic scaling to better show differences
                    const logValue = Math.log10(value + 1);
                    const maxLogValue = Math.log10(maxValue + 1);
                    radius = minSize + (logValue / maxLogValue) * (maxSize - minSize);
                }
                radius = Math.max(minSize, Math.min(maxSize, radius));

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
                    color: color,
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                });

                // Add popup with enhanced information
                const popupContent = `
                    <div style="font-family: Arial, sans-serif; max-width: 250px;">
                        <b style="font-size: 14px;">${item.name || item.facility_id || 'Unknown'}</b><br>
                        <hr style="margin: 5px 0;">
                        <b>Fuel Type:</b> ${item.fuel_type || 'N/A'}<br>
                        <b>Region:</b> ${item.network_region || 'N/A'}<br>
                        <b>Power Output:</b> ${item.power !== null ? item.power.toFixed(2) + ' MW' : 'N/A'}<br>
                        <b>Emissions:</b> ${item.emissions !== null ? item.emissions.toFixed(2) + ' tCO2e' : 'N/A'}<br>
                        <b>Coordinates:</b> ${item.latitude.toFixed(4)}, ${item.longitude.toFixed(4)}
                    </div>
                `;
                marker.bindPopup(popupContent);

                markerClusterGroup.addLayer(marker);
            });

            console.log('✅ Successfully added', validData.length, 'markers to map');

            // Update legend with fuel types present in filtered data
            updateLegend(validData);
        }

        // Update the legend with fuel types present in the data
        function updateLegend(data) {
            const legendContent = document.getElementById('legend-content');
            if (!legendContent) return;

            // Get unique fuel types in the current filtered data
            const fuelTypes = [...new Set(data.map(item => item.fuel_type).filter(f => f !== null))].sort();

            legendContent.innerHTML = '';

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
            const filteredCount = filteredData.length;

            const timestamp = new Date().toLocaleTimeString();
            let statusText = `Last update: ${timestamp} · ${validFacilities} facilities shown`;

            if (filteredCount < totalFacilities) {
                statusText += ` (filtered from ${totalFacilities} total)`;
            }

            statusElement.textContent = statusText;
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

                record = {
                    "facility_id": safe_value(row.get("facility_id")),
                    "name": safe_value(row.get("name")),
                    "fuel_type": fuel_type,
                    "network_region": network_region,
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
