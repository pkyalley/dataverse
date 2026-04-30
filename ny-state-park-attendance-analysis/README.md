# 🌲 NY State Park Attendance Analysis

> An interactive R-based data analysis project exploring New York State Park visitor trends from 2003 to 2024 — with time series analysis, forecasting, and a full Shiny dashboard.

---

## 📖 What Is This Project?

New York State operates over 200 state parks that collectively host tens of millions of visitors each year. This project uses 20+ years of publicly available attendance data to uncover long-term visitor trends, identify the state's most-visited facilities, and forecast future attendance using ARIMA time series modeling.

The project delivers two polished data products:

1. **A static HTML report** — a narrative walkthrough of the data: cleaning, summary statistics, top facilities, yearly trends, growth rates, and ARIMA forecasting.
2. **An interactive Shiny dashboard** — a full multi-tab web app with real-time filters, interactive charts, a Leaflet map, and exportable PDF reports.

---

## 📁 Project Structure

```
ny-state-park-attendance-analysis/
│
├── app.R                                             # Interactive Shiny dashboard
├── data/
│   ├── raw/
│   │   └── State_Park_Annual_Attendance_Figures_by_Facility___Beginning_2003_20250211.csv
│   └── processed/
│       └── processed_attendance.csv                  # Cleaned dataset snapshot
├── reports/
│   ├── state-park-analysis.Rmd                        # R Markdown source
│   └── state-park-analysis.html                       # Rendered HTML report snapshot
├── requirements.R
└── README.md
```

---

## 📊 The Data

**Source:** [NY Open Data — State Park Annual Attendance Figures by Facility (Beginning 2003)](https://data.ny.gov/Recreation/State-Park-Annual-Attendance-Figures-by-Facility-B/8f3n-xj78)

**Period Covered:** 2003 – 2024

**Records:** Annual attendance figures per park facility, statewide.

---

## 🔍 Key Findings

- **Consistent long-term growth** in statewide attendance over the 20-year period.
- **A small number of flagship parks** (Niagara Falls, Jones Beach, Letchworth) account for a disproportionate share of total visits.
- **Notable disruption in 2020** due to COVID-19 park closures, visible as a sharp dip in the trend.
- **ARIMA forecasting** supports continued growth in the 2–5 year horizon.

---

## 🚀 How to Run This Project

### Prerequisites

Install R (≥ 4.2) and RStudio, then install required packages:

```r
source("requirements.R")
```

### 1. Run the HTML Report

Open `reports/state-park-analysis.Rmd` in RStudio and click **Knit**, or run:

```r
rmarkdown::render("reports/state-park-analysis.Rmd")
```

### 2. Launch the Shiny Dashboard

```r
shiny::runApp("app.R")
```

---

## 🗂️ Dashboard Features

- Year range and facility filters
- Attendance trend & year-over-year growth charts
- Top 10 facilities ranking
- ARIMA attendance forecast
- Interactive Leaflet map
- Random Forest per-facility predictions
- Exportable PDF reports

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **R** | Core language |
| **R Markdown** | Static narrative report |
| **ggplot2** | Static visualizations |
| **plotly** | Interactive dashboard charts |
| **Shiny + bslib** | Web application framework |
| **forecast / auto.arima** | Time series forecasting |
| **leaflet** | Interactive map |
| **randomForest** | Per-facility predictions |
| **DT** | Interactive data tables |

---

## 💡 Why This Matters

Park attendance data is a practical lens into public behavior, tourism economics, and infrastructure demand. This project demonstrates how open government data can support evidence-based decisions in conservation, facility planning, and resource allocation.

---

## 📜 Data License

Data published by the New York State Office of Parks, Recreation and Historic Preservation via [NY Open Data](https://data.ny.gov). Freely available for public use.

---

## 👤 Author

**Prince Peter Yalley**
Data analysis, time series modeling, and Shiny development.

[![GitHub](https://img.shields.io/badge/GitHub-pkyalley-181717?style=flat&logo=github)](https://github.com/pkyalley)
[![Email](https://img.shields.io/badge/Email-yalleyp@clarkson.edu-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:yalleyp@clarkson.edu)
