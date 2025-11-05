# Assignment 2 – Australian Electricity Data Pipeline

This repository implements the end-to-end workflow required for Assignment 2: collecting, caching, streaming, and visualising Australian electricity generation and emissions data from the OpenElectricity platform.

> **Important:** The code is designed for the OpenElectricity REST API but requires a valid API key and facility metadata produced in Assignment 1. All credentials must be supplied via environment variables or `config.yml` before running the pipeline.

## Project Structure

```
src/electricity_pipeline/   Core Python package
├── api_client.py           REST client for OpenElectricity
├── assignment1_facilities.py  Builds geocoded facility metadata from Assignment 1
├── caching.py              Helpers for cache paths and persistence
├── config.py               Config loaders and dataclasses
├── dashboard.py            Dash-powered MQTT subscriber map
├── data_processing.py      Normalisation, pivoting, enrichment routines
├── main.py                 Continuous retrieval + MQTT publishing loop
├── publisher.py            MQTT publishing helper
├── retrieval.py            Data retrieval orchestration
└── subscriber.py           MQTT subscription utilities
```

Supporting assets:

- `config.yml` – default configuration (override via env vars for secrets).
- `data/facilities_metadata.csv` – **placeholder** produced during Assignment 1; replace with your authoritative file.
- `cache/` – persisted consolidated datasets (created at runtime).
- `docs/` – place final report and supplementary documentation here.

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide configuration details:
   - Set `OPEN_ELECTRICITY_API_KEY` in the environment _or_ edit `config.yml`.
   - If the API expects a header other than `Authorization: Bearer …`, adjust `api.api_key_header` and `api.api_key_prefix` in `config.yml` (e.g., `x-api-key` with an empty prefix).
   - Update MQTT broker parameters if you are not using a local broker.
   - Integrate Assignment 1 data: keep the downloaded CER CSVs under `data/` and run `python -m electricity_pipeline.assignment1_facilities` to build the canonical `data/facilities_metadata.csv` (with precise geocodes + fuel/region metadata). The script caches geocoding responses (`cache/geocode_cache.json`) and falls back to state centroids if a lookup fails.

## Usage

### Task 1–3: Retrieval, Integration, Publishing

Run the continuous pipeline (defaults to one iteration per minute) for a one-week window in October 2025:

```bash
python -m electricity_pipeline.main \
  --start 2025-10-01 \
  --end 2025-10-08 \
  --metrics power emissions price demand
```

- Set `--iterations` to a positive integer to stop after a fixed number of cycles (useful for testing).
- Pass `--disable-cache` to force a fresh API pull even when cached data is available.
- The script enforces a 60 second pause between iterations and emits MQTT messages with a 0.1 second gap exactly as required.
- **Task 3**: Only records with both power AND emissions are published as combined messages to MQTT, ensuring ordered data stream in event time order.

Consolidated CSV files are stored under `cache/` in the format `YYYY-MM-DD_YYYY-MM-DD_consolidated.csv`. Reuse these for downstream analysis to avoid redundant API calls.

### Task 4: MQTT Subscription and Map Visualisation

Launch the Dash dashboard, which subscribes to the same MQTT topic and renders the **same Folium/MarkerCluster visualization** you built in Assignment 1 (color-coded by fuel type, with rich popups). Filters update in real time as new messages arrive.

```bash
python -m electricity_pipeline.dashboard
```

Navigate to `http://127.0.0.1:8050` (configurable via `config.yml`) to interact with the map. Use the toggles to switch between power output and emissions, and filters to drill into regions or fuel types. **Task 4**: Clicking a marker reveals a popup with the station's name, type (fuel technology), latest power production, and latest emissions data. The dashboard dynamically adds markers as MQTT messages are received, simulating an ordered data stream.

### Task 5: Continuous Execution

The main pipeline script (`electricity_pipeline.main`) loops indefinitely by default (when `--iterations` is 0 or not set) with a 60-second delay between API data retrieval rounds. This delay is in addition to the 0.1-second delay between publishing individual MQTT messages, satisfying the continuous execution requirement. Use `CTRL+C` to terminate.

### Task 6: Documentation & Reporting

- Document findings, challenges, and future improvements in `docs/REPORT.md` (template provided below).
- Log files and console output can be redirected for audit trails if desired.

## Testing Tips

- Start a local MQTT broker (e.g., Mosquitto) before running the publisher or dashboard.
- If geocoding takes too long (Nominatim throttling), rerun `python -m electricity_pipeline.assignment1_facilities --skip-geocode`; it will reuse cached coordinates/state medians.
- Use `--iterations 1` during early testing to avoid long waits.
- If the OpenElectricity API enforces tight rate limits, narrow the facility list or shorten the date window while debugging.

## Troubleshooting

- **Missing facility metadata** – ensure `data/facilities_metadata.csv` exists with the required columns.
- **HTTP errors** – check API key, network connectivity, and API endpoint configuration.
- **MQTT connection issues** – confirm broker credentials, topic names, and firewall settings.
- **Dashboard blank map** – verify that the publisher is running and that MQTT messages are flowing to the topic.

## Next Steps

- Complete `docs/REPORT.md` with your analysis.
- Add automated tests or linting as needed.
- Containerise the solution for reproducible deployment if appropriate.
