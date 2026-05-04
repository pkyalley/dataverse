# NHE Analysis Project

This folder contains a CMS National Health Expenditure analysis notebook, a Streamlit dashboard, source data, and saved outputs.

## Folder Structure

- `data/` - CMS NHE24 source files
- `output/` - generated charts, summary tables, and exported CSV files
- `nhe-analysis.ipynb` - notebook version of the analysis
- `nhe_dashboard.py` - modern dashboard for interactive review

## How to Run

1. Open the notebook and run the cells from top to bottom.
2. Run the dashboard with:

```bash
streamlit run nhe_dashboard.py
```

## What the Notebook Produces

- Saved figures in `output/`
- Key summary CSV files in `output/`
- A reusable project README for quick setup and reuse

## Data Sources

The notebook reads the CMS NHE24 summary file and the Table 02, Table 03, and Table 07 workbooks from `data/`.
