COVID-19 Data Analysis & Interactive Dashboard
===============================================

Overview
--------
This project analyzes a cleaned COVID-19 dataset using Python. It covers
confirmed cases, deaths, and recoveries across countries and over time.
An interactive Streamlit dashboard allows users to explore trends, view
key statistics, and visualize the global spread of COVID-19.

Project Files
-------------
- covid_19_clean_complete.csv    : Cleaned COVID-19 dataset
- covid_19_clean_complete.ipynb  : Jupyter Notebook with data exploration,
                                   visualization, and feature engineering
- dashboard.py                   : Streamlit interactive dashboard
- dashboard_instructions.txt    : Additional instructions for running the dashboard
- readme.txt                     : This file

Dashboard Features
------------------
- Country selector to filter data by region
- KPI metrics: total confirmed cases, deaths, and recoveries
- Line chart showing cumulative trends over time
- Bar chart of daily new confirmed cases
- Interactive world map of global COVID-19 spread
- Date range filter in the sidebar
- Expandable raw data table

Requirements
------------
Make sure the following Python packages are installed before running the dashboard:
  - streamlit
  - pandas
  - plotly

Install them with:
  pip install streamlit pandas plotly

How to Run the Dashboard
------------------------
1. Open a terminal in your code editor (e.g., VS Code or PyCharm).
2. Navigate to the project directory:
     cd path/to/covd_19_data_analysis
3. Run the Streamlit app:
     streamlit run dashboard.py
4. The dashboard will open automatically in your default web browser.
