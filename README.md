# 🌌 DataVerse

> A collection of data analysis, data science, and machine learning projects spanning public policy, finance, healthcare, and business analytics.

---

## 📂 Projects

| Project | Description | Tools |
|---------|-------------|-------|
| [🦠 COVID-19 Data Analysis](./covid-19-data-analysis/) | Global COVID-19 trend analysis with EDA, predictive modeling, and an interactive Streamlit dashboard | Python, Pandas, Streamlit |
| [📈 Gold / USD Price Tracker](./gold-usd-price-tracker/) | Automated collection of historical Gold/USD OHLC price data from Yahoo Finance, saved as Excel | Python, yfinance, Pandas |
| [📉 Telco Customer Churn Analysis](./telco-customer-churn-analysis/) | End-to-end churn prediction with EDA, machine learning, and dual dashboards in Python and R | Python, Scikit-learn, R, Shiny |
| [🌲 NY State Park Attendance Analysis](./ny-state-park-attendance-analysis/) | 20-year visitor trend analysis for NY State Parks with ARIMA forecasting and an interactive Shiny dashboard | R, Shiny, ggplot2, forecast |
| [🏔️ NY Licensed Guides Analysis](./ny-licensed-guides-analysis/) | Geographic and activity analysis of 2,516 licensed outdoor guides in New York State | R, Shiny, plotly |
| [📊 NY Labor Market Analysis](./ny-labor-market-analysis/) | Time series analysis of NY employment and unemployment from the 2008 Financial Crisis through 2019, with ARIMA forecasting | R, Shiny, forecast |

---

## 📌 Repository Conventions

- **Data layout:** each project uses `data/raw/` for source data and `data/processed/` for derived outputs.
- **Reports:** R Markdown reports live in `reports/` and render HTML snapshots committed to the repo.
- **Entrypoints:** apps use `app.R` (Shiny) or `app.py` (Streamlit) at the project root.
- **Dependencies:** Python projects include `requirements.txt`; R projects include `requirements.R`.

---

## 🔧 Tech Stack

This repository uses a mix of tools depending on the project:

- **Python** — Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, yfinance
- **R** — ggplot2, Shiny, plotly, tidyverse, forecast, leaflet, R Markdown

---

## 👤 Author

**Prince Peter Yalley**

[![GitHub](https://img.shields.io/badge/GitHub-pkyalley-181717?style=flat&logo=github)](https://github.com/pkyalley)
[![Email](https://img.shields.io/badge/Email-yalleyp@clarkson.edu-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:yalleyp@clarkson.edu)
