# Assignment 2 Report – Australian Electricity Data Pipeline

## 1. Overview

- **Objective:** Summarise the end-to-end workflow for collecting, integrating, streaming, and visualising OpenElectricity data.
- **Team Members:** _Add names here_
- **Timeframe Covered:** _Document the exact date range retrieved (at least one week in Oct 2025)._

## 2. Data Retrieval

- **API Endpoints:** Document the endpoints used for facilities, metrics, and optional data (price/demand).
- **Access & Authentication:** Note how the API key is stored and injected.
- **Request Strategy & Rate Limiting:** Describe batching, pagination, or throttling used to stay within 500 requests/day.
- **Issues Encountered:** e.g., missing data, downtime, unexpected schema changes.

## 3. Data Integration & Caching

- **Cleaning Steps:** Describe handling of missing or inconsistent values, type conversions, or deduplication.
- **Consolidation:** Explain how power, emissions, and optional metrics were merged.
- **Caching:** Show the location of cached CSV files and justify any storage strategy.

## 4. Streaming via MQTT

- **Topic Design:** Specify the topic hierarchy, QoS level, and payload schema.
- **Publishing Strategy:** Confirm 0.1 s message spacing and message ordering.
- **Broker Configuration:** Document host/port, authentication, and durability settings (if any).

## 5. Subscription & Visualisation

- **Framework:** Detail the dashboard technology (Dash + Plotly).
- **Real-Time Updates:** Explain how MQTT messages are ingested and reflected on the map.
- **User Features:** Filters, toggles, tooltips, and pop-ups.

## 6. Continuous Execution

- **Scheduler Implementation:** Describe the 60-second loop and how the pipeline handles restarts.
- **Monitoring:** Note any logging, alerting, or health checks in place.

## 7. Key Insights & Findings

- Highlight any notable trends in October 2025 data (e.g., peak emissions periods, wind vs. solar performance).
- Include visual summaries or statistics produced from the consolidated dataset.

## 8. Challenges & Resolutions

- Summarise major blockers and how the team resolved them (e.g., API schema changes, broken facilities metadata).

## 9. Recommendations & Future Work

- Suggestions for scaling, reliability improvements, advanced analytics, or UX enhancements for the dashboard.

## 10. Appendix

- **Execution Logs:** Link to relevant logs or monitoring dashboards.
- **Configuration:** Reference config files, environment variables, and secrets management approach.
- **Data Dictionary:** Provide definitions for key columns in the consolidated dataset.

