# 🦠 COVID-19 Data Analysis & Interactive Dashboard

> An end-to-end Python data analysis project exploring global COVID-19 trends — confirmed cases, deaths, and recoveries — with an interactive Streamlit dashboard.

---

## 📖 What Is This Project?

This project analyzes a cleaned global COVID-19 dataset to uncover patterns in the spread and impact of the pandemic. The analysis covers confirmed cases, fatalities, and recoveries across countries and over time, combining exploratory data analysis, feature engineering, and predictive modeling into a single workflow.

Key questions this project answers:
- How did confirmed cases, deaths, and recoveries evolve over time?
- Which countries were hardest hit?
- What trends and inflection points are visible in the data?

---

## 📁 Project Structure

```
covid-19-data-analysis/
│
├── app.py                            # Streamlit interactive dashboard
├── covid_19_clean_complete.ipynb     # Jupyter Notebook — full analysis
├── data/
│   ├── processed/
│   │   └── covid_19_clean_complete.csv
│   └── raw/
│       └── .gitkeep
├── dashboard_instructions.txt        # Instructions for running the dashboard
├── requirements.txt
└── README.md
```

---

## 📊 The Data

**Dataset:** `data/processed/covid_19_clean_complete.csv` — a pre-cleaned version of the Johns Hopkins COVID-19 dataset.

**Key columns:**

| Column | Description |
|--------|-------------|
| `Date` | Date of the record |
| `Country/Region` | Country or territory |
| `Confirmed` | Cumulative confirmed cases |
| `Deaths` | Cumulative deaths |
| `Recovered` | Cumulative recoveries |
| `Active` | Currently active cases |

---

## 🔍 What the Analysis Covers

1. **Data Exploration** — summary statistics, missing value checks, data types
2. **Visualization** — time series plots for cases, deaths, and recoveries
3. **Feature Engineering** — derived metrics (mortality rate, recovery rate, active cases)
4. **Trend Analysis** — identifying peaks, waves, and turning points
5. **Predictive Modeling** — basic forecasting of case trajectories
6. **Excel Export** — final results saved as a structured Excel report

---

## 🚀 How to Run This Project

### Prerequisites

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 1. Run the Jupyter Notebook

Open `covid_19_clean_complete.ipynb` in Jupyter Lab or Jupyter Notebook:

```bash
jupyter notebook covid_19_clean_complete.ipynb
```

### 2. Launch the Streamlit Dashboard

Open a terminal, navigate to this folder, and run:

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **Pandas** | Data manipulation and cleaning |
| **Matplotlib / Seaborn** | Static visualizations |
| **Streamlit** | Interactive web dashboard |
| **Jupyter Notebook** | Exploratory analysis |
| **openpyxl** | Excel export |

---

## 👤 Author

**Prince Peter Yalley**
Data analysis, visualization, and dashboard development.

[![GitHub](https://img.shields.io/badge/GitHub-pkyalley-181717?style=flat&logo=github)](https://github.com/pkyalley)

