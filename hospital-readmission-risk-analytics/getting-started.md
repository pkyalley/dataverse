# Getting Started Guide
## Hospital Readmission Risk Analytics

### Step 1: Install Dependencies
Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Jupyter Notebook
```bash
jupyter notebook hospital-readmission-analysis.ipynb
```

**What the notebook does:**
- 📂 Section 1: Sets up project folders
- 📚 Section 2: Imports all libraries
- 📊 Section 3: Loads CSV data from `data/` folder
- 🔍 Section 4: Creates 9-panel exploratory analysis chart → Saves to `output/`
- 🎯 Section 5: Feature selection and multicollinearity checks
- 🤖 Section 6: Trains logistic regression model
- 📈 Section 7: Displays model coefficients and odds ratios
- ✅ Section 8: Creates 6-panel model validation chart → Saves to `output/`
- ⚠️ Section 9: Applies risk scores to all patients
- 👥 Section 10: Generates high-risk patient queue
- 📊 Section 11: Displays final summary
- 💾 Section 12: Saves processed data and model artifacts

**Expected execution time: 2-3 minutes**

### Step 3: Launch Streamlit Dashboard
After the notebook completes, run:
```bash
streamlit run dashboard-app.py
```

The dashboard will open at: `http://localhost:8501/`

### Dashboard Pages:
1. **📈 Overview** - Key metrics and charts
2. **👥 Patient Explorer** - Filter and download patient data
3. **📉 Risk Distribution** - Risk score analysis
4. **🔍 Model Performance** - Feature importance and odds ratios
5. **📊 Visualizations** - View all generated charts

---

### File Organization After Running:

**Input Data** (`data/` folder):
- ✅ hospital-readmission-dataset.csv
- ✅ high-risk-patient-queue.csv

**Output Files** (`output/` folder after running notebook):
- 📊 01-exploratory-data-analysis.png
- 📊 02-model-validation.png
- 👥 high-risk-patient-queue.csv
- 🤖 model-artifacts.pkl

**Generated Data** (`data/` folder after running notebook):
- 📋 hospital-readmission-with-scores.csv

---

### 🎯 Key Features:

✅ **Images Displayed & Saved**
- All visualizations show inline in Jupyter
- All images automatically save to `output/` folder at 150 DPI

✅ **Organized Data Structure**
- CSV files organized in `data/` folder
- Results saved to `output/` folder
- Clean separation of concerns

✅ **Interactive Dashboard**
- 5 different analytical views
- Real-time filtering of patient data
- Download capabilities
- Model performance metrics

✅ **Professional Output**
- Publication-ready charts
- Comprehensive analysis
- Executive summary metrics
- Detailed documentation

---

### Troubleshooting:

**Q: Images not showing in Jupyter?**
A: Ensure you're running in Jupyter Lab or Jupyter Notebook, not VS Code's notebook viewer.

**Q: "ModuleNotFoundError: No module named 'streamlit'"**
A: Run `pip install streamlit` and make sure you're using the correct Python environment.

**Q: Can't find output files?**
A: Check the `output/` folder after running all notebook cells. They should appear there automatically.

**Q: Dashboard says "Failed to load data"?**
A: Make sure you ran all cells in the Jupyter notebook first to generate the required pickle file.

---

### Next Steps:

1. ✅ Review the generated visualizations in the `output/` folder
2. ✅ Explore patient data in the dashboard
3. ✅ Use the high-risk patient queue for clinical interventions
4. ✅ Monitor model performance metrics
5. ✅ Run notebook again with updated data as needed

---

**For questions or customization, refer to the README.md file.**

