#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Prefer project venv's Python if available; fall back to system python3
PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

# Ensure src/ is importable as a package root
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

START_DATE="${START_DATE:-2025-10-01}"
END_DATE="${END_DATE:-2025-10-08}"
METRICS_STRING="${METRICS:-power emissions price demand}"
ITERATIONS="${ITERATIONS:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
SKIP_METADATA="${SKIP_METADATA:-0}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_DASHBOARD="${SKIP_DASHBOARD:-0}"

IFS=' ' read -r -a METRIC_ARGS <<< "$METRICS_STRING"

run_metadata() {
    if [[ "$SKIP_METADATA" == "1" ]]; then
        echo "[metadata] Skipping metadata build (SKIP_METADATA=1)."
        return
    fi
    echo "[metadata] Building facilities metadata from Assignment 1 datasets..."
    if [[ "${SKIP_GEOCODE:-0}" == "1" ]]; then
        echo "[metadata] Note: SKIP_GEOCODE=1 (faster, uses cached/state centroids)"
        "$PYTHON" -m electricity_pipeline.assignment1_facilities --skip-geocode
    else
        "$PYTHON" -m electricity_pipeline.assignment1_facilities
    fi
}

run_pipeline() {
    if [[ "$SKIP_PIPELINE" == "1" ]]; then
        echo "[pipeline] Skipping pipeline execution (SKIP_PIPELINE=1)."
        return
    fi
    echo "[pipeline] Starting retrieval + MQTT publishing loop..."
    "$PYTHON" -m electricity_pipeline.main \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --metrics "${METRIC_ARGS[@]}" \
        --iterations "$ITERATIONS" \
        --sleep-seconds "$SLEEP_SECONDS"
}

run_dashboard() {
    if [[ "$SKIP_DASHBOARD" == "1" ]]; then
        echo "[dashboard] Skipping dashboard launch (SKIP_DASHBOARD=1)."
        return
    fi
    echo "[dashboard] Launching Dash/Folium map (Ctrl+C to stop)..."
    "$PYTHON" -m electricity_pipeline.dashboard
}

run_metadata
run_pipeline
run_dashboard

