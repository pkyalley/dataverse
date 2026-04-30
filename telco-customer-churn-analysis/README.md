# 📉 Telco Customer Churn Analysis & Interactive Dashboards

> An end-to-end analysis of telecom customer churn using Python and R — combining exploratory data analysis, machine learning, and interactive dashboards built with Streamlit and Shiny.

---

## 📖 What Is This Project?

Customer churn is one of the most costly challenges in the telecom industry. This project takes IBM's Telco Customer Churn dataset and turns it into a full analytical workflow: from raw data to predictive models and interactive dashboards that surface actionable business insights.

**Key questions this analysis answers:**
- Which customer segments are most likely to churn?
- What contract, billing, and service factors drive churn?
- How accurately can we predict whether a customer will leave?

---

## 📁 Project Structure

```
telco-customer-churn-analysis/
│
├── app.py                                   # Streamlit interactive dashboard
├── telco_customer_churn_analysis_code.ipynb # Python: EDA, modeling, visualizations
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── reports/
│   ├── telco_customer_churn_analysis_rstudio.Rmd # R: full analysis report
│   └── dashboard_shiny_app.Rmd                    # R Shiny dashboard
├── requirements.txt
├── requirements.R
└── README.md
```

---

## 📊 The Data

**Source:** [IBM Sample Dataset — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Records:** 7,043 customers | 21 features

### Key Features

| Column | Description |
|--------|-------------|
| `customerID` | Unique customer identifier |
| `gender` | Customer gender |
| `tenure` | Months the customer has been with the company |
| `Contract` | Contract type (Month-to-month, One year, Two year) |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount charged |
| `TechSupport` / `OnlineSecurity` | Add-on service subscriptions |
| `Churn` | Target variable — whether the customer churned (Yes/No) |

---

## 🔍 Key Findings

1. **Month-to-month contracts drive the most churn.** Customers on monthly contracts churn at significantly higher rates than those on annual or two-year plans.
2. **High monthly charges and short tenure are strong churn predictors.** New customers paying premium prices are the most at-risk segment.
3. **Lack of support services correlates with churn.** Customers without tech support or online security are more likely to leave.
4. **Predictive models achieve strong performance.** Logistic Regression and Random Forest classifiers both deliver reliable churn predictions.

---

## 🚀 How to Run This Project

### Python (EDA + Streamlit Dashboard)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Jupyter Notebook for the full analysis:

```bash
jupyter notebook telco_customer_churn_analysis_code.ipynb
```

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

### R (R Markdown Report + Shiny Dashboard)

Install R (≥ 4.2) and RStudio, then install required packages:

```r
source("requirements.R")
```

Render the R Markdown report:

```r
rmarkdown::render("reports/telco_customer_churn_analysis_rstudio.Rmd")
```

Launch the Shiny dashboard:

```r
rmarkdown::run("reports/dashboard_shiny_app.Rmd")
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | EDA, machine learning, Streamlit dashboard |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Static visualizations |
| **Scikit-learn** | Logistic Regression, Random Forest |
| **Streamlit** | Interactive Python dashboard |
| **R / R Markdown** | Narrative analysis report |
| **ggplot2** | Static R visualizations |
| **Shiny** | Interactive R dashboard |
| **dplyr** | Data wrangling in R |

---

## 💡 Why Python and R?

Both tools were used deliberately:

- **Python** is ideal for machine learning pipelines and rapid dashboard prototyping with Streamlit.
- **R and Shiny** excel at statistical visualization and producing polished, publication-ready analytical reports.

Using both languages showcases how modern data science workflows can leverage the strengths of each ecosystem.

---

## 👤 Author

**Prince Peter Yalley**
Data analysis, machine learning, and interactive dashboard development.

[![GitHub](https://img.shields.io/badge/GitHub-pkyalley-181717?style=flat&logo=github)](https://github.com/pkyalley)
[![Email](https://img.shields.io/badge/Email-yalleyp@clarkson.edu-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:yalleyp@clarkson.edu)
