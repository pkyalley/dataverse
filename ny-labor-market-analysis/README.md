# 📉 NY Labor Market Unemployment Analysis

> A time series analysis of employment, unemployment, and labor force participation in New York State — from the 2008 Financial Crisis through 2019, with ARIMA-based forecasting.

---

## 📖 What Is This Project?

Between 2007 and 2019, the U.S. labor market went through one of the most dramatic cycles in modern economic history — a near-collapse during the 2008 Financial Crisis, followed by one of the slowest recoveries on record.

This project takes monthly labor market data and turns it into two polished, interactive data products:

1. **An HTML Report** — a narrative-driven analysis with publication-quality charts, written in R Markdown.
2. **An Interactive Shiny Dashboard** — a multi-tab web application with real-time filters, interactive plots, and a 24-month ARIMA forecast.

**Key questions this analysis answers:**

- How did the 2008 Financial Crisis reshape the labor market, and how long did recovery actually take?
- Is there a seasonal pattern to unemployment throughout the year?
- What does the Labor Force Participation Rate reveal that the unemployment rate hides?
- Based on historical trends, where was unemployment structurally headed after 2019?

---

## 📁 Project Structure

```
ny-labor-market-analysis/
│
├── app.R                         # Shiny dashboard application
├── data/
│   └── raw/
│       └── Employment__Unemployment__and_Labor_Force_Data.xlsx
├── reports/
│   ├── labor_market_analysis.Rmd     # Narrative HTML report (R Markdown)
│   ├── labor_market_analysis.html    # Rendered HTML report snapshot
│   └── report_style.css              # Custom CSS styling for the report
├── requirements.R
└── README.md
```

---

## 📊 The Data

**Source:** Employment, Unemployment & Labor Force Data (NY Open Data)

**Period:** January 2007 – August 2019 (152 monthly records)

### Key Fields

| Column | Description |
|--------|-------------|
| `Date` | Month and year of the record |
| `Unemployment Rate` | % of labor force that is unemployed |
| `Employment Rate` | % of labor force that is employed |
| `Labor Force Participation Rate` | % of working-age population in the labor force |
| `Employed` | Number of employed persons |
| `Unemployed` | Number of unemployed persons |
| `Civilian Labor Force` | Total labor force size |

---

## 🔍 Key Findings

1. **The 2008 Financial Crisis doubled unemployment in under 2 years.** The rate surged from 3.3% in late 2007 to 7.8% by January 2010 — a ~150% increase.

2. **Recovery was real but painfully slow.** It took until 2015–2016 to return to pre-crisis unemployment levels — one of the slowest labor market recoveries in post-war U.S. history.

3. **The Labor Force Participation Rate tells a darker story.** The LFPR dropped from ~69% in 2008 to under 67% by 2015, meaning part of the unemployment "improvement" came from people giving up on finding work, not from actual job creation.

4. **Employment grew consistently from 2010 onward.** Absolute employment rose every year after 2009 — from ~2.8M to ~3.1M employed persons.

5. **The ARIMA forecast projected continued decline.** Based purely on historical trend, unemployment was on track to approach or breach the 3.3% pre-crisis low by 2020–2021 — before COVID-19 changed everything.

---

## 🛠️ How to Run This Project Locally

### Prerequisites

Install R (≥ 4.2) and RStudio, then install the required packages:

```r
source("requirements.R")
```

### 1. Clone the Repository

```bash
git clone https://github.com/pkyalley/dataverse.git
cd dataverse/ny-labor-market-analysis
```

### 2. Run the HTML Report

Open `reports/labor_market_analysis.Rmd` in RStudio and click **Knit**, or run:

```r
rmarkdown::render("reports/labor_market_analysis.Rmd")
```

### 3. Launch the Shiny Dashboard

Open `app.R` in RStudio and click **Run App**, or run:

```r
shiny::runApp(".")
```

> **Note:** Both the report and the app expect the Excel file to be inside `data/raw/` relative to the project root.

---

## 📸 Screenshots

*(Add screenshots here after running the apps locally.)*

| View | Description |
|------|-------------|
| `screenshot_report.png` | HTML report — timeline and key findings |
| `screenshot_dashboard_overview.png` | Shiny dashboard — Overview tab |
| `screenshot_dashboard_trends.png` | Shiny dashboard — Trend Deep Dive tab |
| `screenshot_dashboard_seasonal.png` | Shiny dashboard — Seasonal Patterns tab |
| `screenshot_dashboard_forecast.png` | Shiny dashboard — ARIMA Forecast tab |

---

## 💡 Why This Project

Unemployment data is one of the most widely cited economic indicators — but it is also one of the most misunderstood. The headline unemployment rate alone hides important dynamics like labor force dropout, seasonal hiring cycles, and structural versus cyclical joblessness.

This project was built to go beyond the headline number and tell the full story of a labor market under stress, in recovery, and on the path toward expansion — using rigorous time series methods and clear visual communication.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **R** | Core language |
| **R Markdown** | Narrative HTML report |
| **ggplot2** | Static visualizations |
| **plotly** | Interactive dashboard charts |
| **Shiny + shinydashboard** | Web application framework |
| **forecast / auto.arima** | Time series forecasting |
| **dplyr / tidyr** | Data manipulation |
| **DT** | Interactive data tables |
| **kableExtra** | Report tables |

---

## 📜 Data License

Data sourced from NY Open Data, published by the New York State Department of Labor. Freely available for public use under the [NY Open Data Terms of Use](https://data.ny.gov/download/77gx-ii52/application%2Fpdf).

---

## 👤 Author

**Prince Peter Yalley**
Data analysis, time series modeling, and Shiny development.

[![GitHub](https://img.shields.io/badge/GitHub-pkyalley-181717?style=flat&logo=github)](https://github.com/pkyalley)
[![Email](https://img.shields.io/badge/Email-yalleyp@clarkson.edu-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:yalleyp@clarkson.edu)

---

*Built with R, ggplot2, and Shiny. Data courtesy of NY Open Data / NY State Department of Labor.*
