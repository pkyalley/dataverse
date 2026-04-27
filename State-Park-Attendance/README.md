# 🌲 State Park Attendance Analytics

Exploring New York State Park visitor trends (2003–2024) through interactive data analysis and forecasting.

---

## What's in this repo

| File | Description |
|------|-------------|
| `state-park-analysis.Rmd` | R Markdown source for the static report |
| `state-park-analysis.html` | Rendered HTML report |
| `app.R` | Interactive Shiny dashboard |
| `processed_attendance.csv` | Cleaned dataset |
| `State_Park_Annual_Attendance_Figures_by_Facility_...csv` | Raw source data |

---

## The project

**Static report** — a step-by-step walkthrough of the dataset: cleaning, summary statistics, top facilities, yearly trends, growth rates, and ARIMA forecasting. Good starting point before touching the dashboard.

**Shiny dashboard** — interactive version of the same analysis. Filter by year range or facility, explore trends, compare facilities, and generate attendance forecasts.

### Dashboard features
- Year range and facility filters
- Attendance trend & year-over-year growth charts
- Top 10 facilities ranking
- ARIMA attendance forecast
- Interactive Leaflet map
- Random Forest per-facility predictions
- Exportable PDF reports

---

## Getting started

```r
# Install dependencies
install.packages(c(
  "shiny", "tidyverse", "plotly", "forecast", "DT",
  "leaflet", "randomForest", "rmarkdown", "bslib", "shinycssloaders"
))

# Run the dashboard
shiny::runApp("app.R")
```

---

## Key findings

- Statewide attendance shows consistent long-term growth across the 20-year period
- A small number of flagship parks (Niagara Falls, Jones Beach, Letchworth) account for a disproportionate share of total visits
- Visitor patterns vary significantly year-over-year, with a notable disruption in 2020
- ARIMA forecasting supports continued growth in the 2–5 year horizon

---

## Tools

R · Shiny · tidyverse · ggplot2 · plotly · leaflet · randomForest · forecast · bslib

---

## Why this matters

Park attendance data is a practical lens into public behavior, tourism economics, and infrastructure demand. This project demonstrates how open government data can support evidence-based decisions in conservation, facility planning, and resource allocation.
