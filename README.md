# 🌌 DataVerse – Exploring the Universe of Data Science & AI 🚀

Welcome to DataVerse, a centralized hub for data analysis, web scraping, machine learning, and AI projects.

## Projects

- **State-Park-Attendance** — R Markdown report + Shiny dashboard for NY state park attendance trends (`reports/`, `app.R`)
- **Who-Guides-New-York** — R Markdown report + Shiny dashboard for NY licensed guides (`reports/`, `app.R`)
- **labour-market-analysis** — R Markdown report + Shiny dashboard for NY labor market trends (`reports/`, `app.R`)
- **telco_customer_churn_analysis** — Streamlit dashboard + R Markdown analysis for churn (`app.py`, `reports/`)
- **covd_19_data_analysis** — Streamlit dashboard + notebook for COVID-19 trends (`app.py`, notebook)
- **gold_usd_prices_from_yf** — Python data collection for Gold/USD OHLC data (script)

## Repository conventions

- **Data layout:** each project uses `data/raw/` for source data and `data/processed/` for derived outputs.
- **Reports:** R Markdown reports live in `reports/` and render HTML snapshots committed to the repo.
- **Entrypoints:** apps use `app.R` (Shiny) or `app.py` (Streamlit) at the project root.
- **Dependencies:** Python projects include `requirements.txt`; R projects include `requirements.R`.
