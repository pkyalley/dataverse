HRRP Dashboard

This Streamlit app provides interactive exploration of the FY 2026 HRRP dataset.

Setup

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the dashboard:

```powershell
streamlit run dashboard.py
```

Notes

- The app expects `FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv` to be in the same folder.
- Figures are displayed inline; the original notebook still saves images to `outputs/` when executed.

Data source and context
-----------------------

Data: https://data.cms.gov/provider-data/dataset/9n3s-kdb3

Released: February 25, 2026

Last Modified: January 26, 2026

In October 2012, CMS began reducing Medicare payments for subsection(d) hospitals with excess readmissions under the Hospital Readmissions Reduction Program (HRRP). Excess readmissions are measured by a ratio, calculated by dividing a hospital's predicted rate of readmissions for heart attack (AMI), heart failure (HF), pneumonia, chronic obstructive pulmonary disease (COPD), hip/knee replacement (THA/TKA), and coronary artery bypass graft surgery (CABG) by the expected rate of readmissions, based on an average hospital with similar patients.
