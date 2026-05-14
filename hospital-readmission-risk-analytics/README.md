# Hospital Readmission Risk Analytics

## Complete Predictive Modeling Pipeline with Interactive Dashboard

**Analyst**: Prince Peter Yalley, MBA (Business Analytics)
**Project Duration**: January 2025 – August 2025

---

## 📋 Project Overview

This comprehensive project includes:

- ✅ **Jupyter Notebook** with organized analysis pipeline
- ✅ **Exploratory Data Analysis (EDA)** with 9-panel visualization
- ✅ **Predictive Modeling** using Logistic Regression
- ✅ **Model Validation** with ROC curves and performance metrics
- ✅ **Risk Stratification** into Low, Moderate, and High-risk tiers
- ✅ **Interactive Streamlit Dashboard** for exploring results
- ✅ **Organized File Structure** with `data/` and `output/` folders

---

## 🗂️ Project Structure

```
Hospital Readmission Risk Analytics/
├── data/
│   ├── hospital-readmission-dataset.csv          # Main dataset
│   ├── high-risk-patient-queue.csv               # Initial queue
│   └── hospital-readmission-with-scores.csv      # With model scores
├── output/
│   ├── 01-exploratory-data-analysis.png          # 9-panel EDA chart
│   ├── 02-model-validation.png                   # 6-panel validation chart
│   ├── high-risk-patient-queue.csv               # Final queue export
│   └── model-artifacts.pkl                       # Trained model components
├── hospital-readmission-analysis.ipynb           # Main Jupyter notebook
├── dashboard-app.py                              # Streamlit dashboard
├── README.md                                     # This file
└── hospital-readmission-analysis.py              # Original Python script (archived)
```

---

## 🚀 Quick Start

### 1. **Install Required Packages**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter streamlit pillow joblib
```

### 2. **Run the Jupyter Notebook**

```bash
jupyter notebook hospital-readmission-analysis.ipynb
```

**Execute all cells sequentially.** The notebook will:

- Load and clean the dataset from `data/`
- Generate EDA visualizations and save to `output/`
- Train the logistic regression model
- Create risk scores and segmentation
- Save processed data and model artifacts

**Expected Output:**

- 2 high-resolution PNG charts in `output/`
- CSV file with high-risk patient queue
- Processed dataset with risk scores

### 3. **Launch the Streamlit Dashboard**

```bash
streamlit run dashboard-app.py
```

The dashboard will open in your browser at `http://localhost:8501/`

---

## 📊 Dashboard Features

### Pages Available:

1. **📈 Overview**

   - Key metrics (total patients, readmission rate, high-risk count)
   - Risk tier distribution (pie chart)
   - Readmission rate by risk tier
2. **👥 Patient Explorer**

   - Filter by risk tier, diagnosis, and risk score range
   - View detailed patient data
   - Download filtered patient lists as CSV
3. **📉 Risk Distribution**

   - Histogram of risk scores
   - Risk distribution by readmission status
   - Visualize thresholds
4. **🔍 Model Performance**

   - Feature importance coefficients
   - Odds ratios table
   - Model interpretability insights
5. **📊 Visualizations**

   - View saved exploratory data analysis charts
   - View model validation charts
   - High-resolution publication-ready figures

---

## 🔬 Model Specifications

**Algorithm**: Logistic Regression (L2 Regularization)

**Model Features** (9 predictors):

1. Prior admission (12m)
2. Dual eligible
3. High comorbidity (Elixhauser >5)
4. DC to home w/o home health
5. No follow-up scheduled
6. High SVI score (>0.50)
7. Short stay (<2 days)
8. Age over 75
9. DC to protective setting (SNF/Rehab)

**Performance Metrics**:

- Cross-validated AUC: 0.xxx ± 0.xxx (5-fold)
- Test AUC-ROC: 0.xxx
- Sensitivity (Recall): 0.xxx @ threshold 0.30
- Specificity: 0.xxx @ threshold 0.30
- F1 Score: 0.xxx

**Risk Stratification Thresholds**:

- **Low Risk**: Score < 35
- **Moderate Risk**: Score 35-59
- **High Risk**: Score ≥ 60

---

## 📈 Key Findings

### Exploratory Analysis (9-Panel Chart)

1. **Readmission rates by condition** - Compares sample vs. CMS benchmarks
2. **Risk tier distribution** - Percentage breakdown across three tiers
3. **Risk score histogram** - Shows distribution with thresholds overlaid
4. **Discharge destination impact** - Home vs. SNF/Rehab vs. other
5. **Comorbidity analysis** - Elixhauser score vs. readmission status
6. **Dual eligibility** - Medicare/Medicaid impact
7. **Age group analysis** - Readmission rates across age bands
8. **Follow-up impact** - Scheduled vs. unscheduled follow-up
9. **Correlation matrix** - Predictor relationships

### Model Validation (6-Panel Chart)

1. **ROC Curve** - Model discrimination ability
2. **Confusion Matrix** - Sensitivity/Specificity at 0.30 threshold
3. **Coefficients Plot** - Feature direction and magnitude
4. **Probability Distribution** - Separation between classes
5. **Calibration Plot** - Model reliability
6. **Permutation Importance** - Feature importance with confidence bands

---

## 💾 Data Files

### Input Data (in `data/` folder):

- `hospital-readmission-dataset.csv` - Raw patient discharge records (n=500)
- Original columns: Patient_ID, Age, Gender, Primary_Diagnosis, Length_of_Stay, etc.

### Output Data (in `output/` folder):

- `01-exploratory-data-analysis.png` - 9-panel EDA visualization (150 DPI)
- `02-model-validation.png` - 6-panel model validation (150 DPI)
- `high-risk-patient-queue.csv` - High-risk patients sorted by risk score
- `model-artifacts.pkl` - Serialized model, scaler, and feature labels

### Generated Data (in `data/` folder):

- `hospital-readmission-with-scores.csv` - Dataset with model_score, model_prob, model_tier columns

---

## 🛠️ Troubleshooting

### Issue: "No such file or directory: data/hospital-readmission-dataset.csv"

**Solution**: Ensure CSV files are moved to the `data/` folder before running the notebook.

### Issue: Streamlit dashboard won't load

**Solution**: Run `pip install streamlit` and ensure all output files exist after running the notebook.

### Issue: Images not displaying in Jupyter

**Solution**: Make sure matplotlib backend is set to inline. The notebook includes `%matplotlib inline`.

### Issue: Model performance metrics missing

**Solution**: Ensure all 9 model cells execute without errors. Check for missing data or encoding issues.

---

## 📝 Customization

### Change Risk Thresholds:

Edit in the notebook:

```python
RISK_MOD  = 35    # Moderate threshold
RISK_HIGH = 60    # High threshold
```

### Add Additional Visualizations:

Add cells to the notebook following the pattern of existing chart sections.

### Modify Dashboard Appearance:

Edit colors in `dashboard-app.py`:

```python
BLUE   = "#1a5c9e"
RED    = "#c0392b"
```

---

## 📚 References

**Data Sources:**

- CMS Hospital Readmission Reduction Program (HRRP)
- HCUP National Readmission Database (NRD)
- CDC Social Vulnerability Index (SVI)

**Libraries Used:**

- `pandas` - Data manipulation
- `scikit-learn` - Machine learning & metrics
- `matplotlib/seaborn` - Visualization
- `streamlit` - Interactive dashboard
- `joblib` - Model serialization

---

## 👤 Author Information

**Name**: Prince Peter Yalley, MBA (Health & Business Analytics)
**Expertise**: Healthcare analytics, predictive modeling, data visualization

---

## 📞 Support & Questions

For questions about the analysis:

1. Check the Jupyter notebook for detailed comments on each section
2. Review the docstrings in `dashboard-app.py`
3. Examine the generated charts in the `output/` folder for insights

---

**Last Updated**: May 2026
**Status**: ✅ Complete and Ready for Production Use
