# 📈 Gold / USD Price Tracker

> A Python script that fetches historical OHLC (Open, High, Low, Close) price data for Gold/USD from Yahoo Finance and saves it as a structured Excel file for further analysis.

---

## 📖 What Is This Project?

Gold is one of the most widely tracked financial assets in the world — a benchmark for inflation hedging, safe-haven demand, and macroeconomic sentiment. This project automates the collection of daily Gold/USD price data using Yahoo Finance's public API and organizes it into a clean, analysis-ready Excel file.

The workflow is simple, reproducible, and easily adaptable to other commodities, currencies, or time ranges.

---

## 📁 Project Structure

```
gold-usd-price-tracker/
│
├── fetch_gold_usd_prices_from_yf.py   # Python script to fetch and save data
├── data/
│   ├── processed/
│   │   └── Gold_USD_Prices_from_yf.xlsx  # Output Excel file (Jan 2023 – Dec 2024)
│   └── raw/
│       └── .gitkeep
├── requirements.txt
└── README.md
```

---

## 📊 The Data

**Source:** Yahoo Finance via the `yfinance` Python library

**Ticker Symbol:** `GC=F` (Gold Futures — standard Yahoo Finance proxy for Gold/USD)

**Period Covered:** January 2023 – December 2024

**Output Format:** Excel (`.xlsx`)

### Columns

| Column | Description |
|--------|-------------|
| `Date` | Trading date |
| `Open` | Opening price (USD) |
| `High` | Daily high price (USD) |
| `Low` | Daily low price (USD) |
| `Close` | Closing price (USD) |

---

## 🚀 How to Run

### Prerequisites

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Run the script

```bash
python fetch_gold_usd_prices_from_yf.py
```

This will fetch the data and save it as `data/processed/Gold_USD_Prices_from_yf.xlsx`.

---

## ⚙️ Customization

To change the date range or asset, edit these lines in the script:

```python
stock_symbol = "GC=F"       # Yahoo Finance ticker (e.g. "GC=F" for Gold, "SI=F" for Silver)
start_date = "2023-01-01"   # Start date (YYYY-MM-DD)
end_date   = "2024-12-31"   # End date   (YYYY-MM-DD)
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **yfinance** | Yahoo Finance data fetching |
| **Pandas** | Data processing |
| **openpyxl** | Excel file export |

---

## 👤 Author

**Prince Peter Yalley**
Data collection, automation, and financial data analysis.

[![GitHub](https://img.shields.io/badge/GitHub-pkyalley-181717?style=flat&logo=github)](https://github.com/pkyalley)
