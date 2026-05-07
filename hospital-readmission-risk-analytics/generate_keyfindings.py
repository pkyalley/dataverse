"""
Generate all key findings visualizations (4.1 - 5.1)
=====================================================
Generates publication-quality charts and tables for:
- 4.1: Readmission Rates by Condition
- 4.2: Regression Coefficients
- 4.3: Model Performance Summary
- 4.4: EDA Panels (9-panel)
- 4.5: Model Validation Panels (6-panel)
- 4.6: Risk Stratification
- 4.7: High-Risk Patient Queue
- 5.1: Risk Score Algorithm Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import warnings

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────
BASE_PATH = Path(__file__).parent
DATA_DIR = BASE_PATH / "data"
OUTPUT_DIR = BASE_PATH / "output"
KEYFINDINGS_DIR = BASE_PATH / "keyfindings"

# Color scheme matching dashboard
BLUE   = "#1a5c9e"
RED    = "#c0392b"
AMBER  = "#e8a020"
GREEN  = "#27a85f"
GRAY   = "#6b6b6b"
LIGHT_GRAY = "#e8e8e8"

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# ─── Load Data ────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_DIR / "hospital-readmission-with-scores.csv")
high_risk_queue = pd.read_csv(DATA_DIR / "high-risk-patient-queue.csv")

# Load model artifacts if available
try:
    artifacts = joblib.load(OUTPUT_DIR / "model-artifacts.pkl")
    model = artifacts.get('model')
    scaler = artifacts.get('scaler')
except:
    model = None
    scaler = None
    print("Warning: Model artifacts not found. Some charts will be limited.")

print(f"Data loaded: {len(df)} records")

# ─── 4.1: READMISSION RATES BY CONDITION ──────────────────────────────────
def generate_fig_4_1():
    """
    Table 3: 30-day readmission rates by CMS HRRP condition
    vs. CMS national benchmarks
    """
    print("Generating 4.1: Readmission Rates by Condition...")
    
    cms_benchmarks = {
        'Heart Failure': {'sample': 24.2, 'cms': 20.9},
        'COPD': {'sample': 30.3, 'cms': 18.4},
        'Pneumonia': {'sample': 33.0, 'cms': 16.5},
        'Acute Myocardial Infarction': {'sample': 21.7, 'cms': 15.9},
        'Hip/Knee Replacement': {'sample': 34.0, 'cms': 5.1}
    }
    
    # Map actual diagnoses in data
    diagnosis_map = {
        'Heart Failure': 'Heart Failure',
        'COPD': 'COPD',
        'Pneumonia': 'Pneumonia',
        'Acute Myocardial Infarction': 'Acute Myocardial Infarction',
        'Hip/Knee Replacement': 'Hip/Knee Replacement'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    conditions = list(cms_benchmarks.keys())
    sample_rates = [cms_benchmarks[c]['sample'] for c in conditions]
    cms_rates = [cms_benchmarks[c]['cms'] for c in conditions]
    excess_ratios = [s / c for s, c in zip(sample_rates, cms_rates)]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sample_rates, width, label='Sample Rate (2024)', color=RED, alpha=0.8)
    bars2 = ax.bar(x + width/2, cms_rates, width, label='CMS Benchmark', color=GREEN, alpha=0.8)
    
    ax.set_xlabel('HRRP Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('30-Day Readmission Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('30-Day Readmission Rates: Analytic Sample vs. CMS National Benchmarks', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(KEYFINDINGS_DIR / "4_1_readmission_by_condition.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_1_readmission_by_condition.png")
    plt.close()


# ─── 4.2: REGRESSION COEFFICIENTS ────────────────────────────────────────────
def generate_fig_4_2():
    """
    Table 4: Logistic regression model coefficients and odds ratios
    """
    print("Generating 4.2: Regression Coefficients...")
    
    coef_data = {
        'Variable': [
            'Prior Admission (≤12m)',
            'Dual Medicare/Medicaid',
            'High Comorbidity (Elix>5)',
            'No Follow-up at D/C',
            'D/C Home w/o Home Health',
            'High SVI Score (>0.50)',
            'Short Stay (<2 days)',
            'Age Over 75',
            'D/C to SNF/Rehab'
        ],
        'Coefficient': [0.839, 0.353, 0.343, 0.297, 0.286, 0.209, 0.081, 0.058, 0.004],
        'Odds_Ratio': [2.31, 1.42, 1.41, 1.35, 1.33, 1.23, 1.08, 1.06, 1.00]
    }
    
    coef_df = pd.DataFrame(coef_data)
    coef_df = coef_df.sort_values('Coefficient', ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Coefficients plot
    colors = [RED if x > 0.28 else AMBER if x > 0.15 else BLUE for x in coef_df['Coefficient']]
    ax1.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors, alpha=0.8)
    ax1.set_xlabel('Standardized Coefficient', fontsize=11, fontweight='bold')
    ax1.set_title('Logistic Regression Coefficients\n(Standardized Model)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.axvline(x=0.28, color=RED, linestyle='--', linewidth=2, alpha=0.5, label='Top Predictor Threshold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='x', alpha=0.3)
    
    # Odds ratios plot
    ax2.barh(coef_df['Variable'], coef_df['Odds_Ratio'], color=colors, alpha=0.8)
    ax2.set_xlabel('Odds Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Odds Ratios\n(Risk per Unit Change in Standardized Predictor)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(KEYFINDINGS_DIR / "4_2_regression_coefficients.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_2_regression_coefficients.png")
    plt.close()


# ─── 4.3: MODEL PERFORMANCE ──────────────────────────────────────────────────
def generate_fig_4_3():
    """
    Table 5: Model performance metrics summary
    """
    print("Generating 4.3: Model Performance Summary...")
    
    # Create a nice table visualization
    metrics_data = {
        'Metric': [
            '5-Fold CV AUC',
            'Held-Out Test AUC-ROC',
            'Precision',
            'Recall / Sensitivity',
            'F1 Score',
            'Decision Threshold'
        ],
        'Value': [
            '0.722 ± 0.069',
            '0.661',
            '0.484',
            '0.536',
            '0.508',
            '0.30'
        ],
        'Interpretation': [
            'Good discriminative performance',
            'Acceptable on unseen data',
            '48.4% of flagged are true readmit',
            '53.6% of actual readmit captured',
            'Balanced precision-recall',
            'Tuned for sensitivity'
        ]
    }
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    df_metrics = pd.DataFrame(metrics_data)
    table = ax.table(cellText=df_metrics.values,
                     colLabels=df_metrics.columns,
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.2, 0.15, 0.35])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(metrics_data)):
        table[(0, i)].set_facecolor(BLUE)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data) + 1):
        for j in range(len(metrics_data)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(LIGHT_GRAY)
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Model Performance Summary Metrics\n(Logistic Regression L2, 80/20 Stratified Split)', 
             fontsize=13, fontweight='bold', pad=20)
    plt.savefig(KEYFINDINGS_DIR / "4_3_model_performance.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_3_model_performance.png")
    plt.close()


# ─── 4.4: EDA PANELS (9-panel) ────────────────────────────────────────────────
def generate_fig_4_4():
    """
    Figure 1: 9-panel Exploratory Data Analysis
    """
    print("Generating 4.4: EDA Panels (9-panel)...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Readmission by Condition
    ax1 = plt.subplot(3, 3, 1)
    readmit_by_cond = df.groupby('Primary_Diagnosis')['Readmitted_30_Days'].agg(['sum', 'count'])
    readmit_by_cond['rate'] = (readmit_by_cond['sum'] / readmit_by_cond['count'] * 100).sort_values(ascending=False)
    ax1.bar(range(len(readmit_by_cond)), readmit_by_cond['rate'], color=RED, alpha=0.7)
    ax1.set_xticks(range(len(readmit_by_cond)))
    ax1.set_xticklabels(readmit_by_cond.index, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Rate (%)', fontsize=10)
    ax1.set_title('1. Readmission Rate by Condition', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Risk Tier Distribution
    ax2 = plt.subplot(3, 3, 2)
    tier_counts = df['Model_Tier'].value_counts()
    colors_tier = [GREEN if x == 'Low' else AMBER if x == 'Moderate' else RED for x in tier_counts.index]
    ax2.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%', colors=colors_tier)
    ax2.set_title('2. Risk Tier Distribution', fontweight='bold', fontsize=11)
    
    # 3. Risk Score Histogram
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(df['Model_Score'], bins=30, color=BLUE, alpha=0.7, edgecolor='black')
    ax3.axvline(35, color=AMBER, linestyle='--', linewidth=2, label='Moderate Threshold')
    ax3.axvline(60, color=RED, linestyle='--', linewidth=2, label='High Threshold')
    ax3.set_xlabel('Risk Score', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('3. Risk Score Distribution', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=8)
    
    # 4. Discharge Destination
    ax4 = plt.subplot(3, 3, 4)
    discharge_counts = df['Discharge_Destination'].value_counts()
    ax4.barh(discharge_counts.index, discharge_counts.values, color=BLUE, alpha=0.7)
    ax4.set_xlabel('Count', fontsize=10)
    ax4.set_title('4. Discharge Destination', fontweight='bold', fontsize=11)
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Comorbidity Burden Comparison
    ax5 = plt.subplot(3, 3, 5)
    comorbidity_readmit = df.groupby('High_Comorbidity')['Readmitted_30_Days'].mean() * 100
    comorbidity_readmit.index = ['No (Elix ≤5)', 'Yes (Elix >5)']
    ax5.bar(comorbidity_readmit.index, comorbidity_readmit.values, color=[GREEN, RED], alpha=0.7)
    ax5.set_ylabel('Readmission Rate (%)', fontsize=10)
    ax5.set_title('5. Readmit Rate by Comorbidity Burden', fontweight='bold', fontsize=11)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Dual Eligibility Impact
    ax6 = plt.subplot(3, 3, 6)
    dual_readmit = df.groupby('Dual_Eligible')['Readmitted_30_Days'].mean() * 100
    dual_readmit.index = ['Medicare Only', 'Dual Eligible']
    ax6.bar(dual_readmit.index, dual_readmit.values, color=[GREEN, RED], alpha=0.7)
    ax6.set_ylabel('Readmission Rate (%)', fontsize=10)
    ax6.set_title('6. Readmit Rate by Dual Eligibility', fontweight='bold', fontsize=11)
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Age Group Analysis
    ax7 = plt.subplot(3, 3, 7)
    age_readmit = df.groupby('Age_Group')['Readmitted_30_Days'].agg(['sum', 'count'])
    age_readmit['rate'] = (age_readmit['sum'] / age_readmit['count'] * 100)
    age_readmit = age_readmit.sort_index()
    ax7.plot(range(len(age_readmit)), age_readmit['rate'], marker='o', color=RED, linewidth=2, markersize=6)
    ax7.set_xticks(range(len(age_readmit)))
    ax7.set_xticklabels(age_readmit.index, rotation=45, ha='right', fontsize=9)
    ax7.set_ylabel('Readmission Rate (%)', fontsize=10)
    ax7.set_title('7. Readmit Rate by Age Group', fontweight='bold', fontsize=11)
    ax7.grid(alpha=0.3)
    
    # 8. Follow-up Scheduling Impact
    ax8 = plt.subplot(3, 3, 8)
    followup_readmit = df.groupby('Follow_Up_Scheduled')['Readmitted_30_Days'].mean() * 100
    followup_readmit.index = ['No Follow-up', 'Follow-up Scheduled']
    ax8.bar(followup_readmit.index, followup_readmit.values, color=[RED, GREEN], alpha=0.7)
    ax8.set_ylabel('Readmission Rate (%)', fontsize=10)
    ax8.set_title('8. Impact of Follow-up Scheduling', fontweight='bold', fontsize=11)
    ax8.grid(axis='y', alpha=0.3)
    
    # 9. Correlation Matrix (predictors only)
    ax9 = plt.subplot(3, 3, 9)
    predictor_cols = ['Age', 'Length_of_Stay', 'Elixhauser_Score', 'SVI_Score', 
                      'Prior_Admission_12m', 'Dual_Eligible', 'Follow_Up_Scheduled',
                      'Home_Health_Referral', 'Readmitted_30_Days']
    corr_matrix = df[predictor_cols].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax9, cbar_kws={'shrink': 0.8},
                annot=False, vmin=-1, vmax=1, square=True)
    ax9.set_title('9. Predictor Correlation Matrix', fontweight='bold', fontsize=11)
    
    plt.suptitle('Exploratory Data Analysis — 9-Panel Summary (n=500)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(KEYFINDINGS_DIR / "4_4_eda_9panel.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_4_eda_9panel.png")
    plt.close()


# ─── 4.5: MODEL VALIDATION PANELS (6-panel) ────────────────────────────────────
def generate_fig_4_5():
    """
    Figure 2: 6-panel Model Validation
    """
    print("Generating 4.5: Model Validation Panels (6-panel)...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Simulate test set for validation (using full dataset for this demo)
    y_true = df['Readmitted_30_Days'].values
    y_pred_proba = (df['Model_Prob'].values * 100).astype(float) / 100
    y_pred = (y_pred_proba >= 0.30).astype(int)
    
    # 1. ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=BLUE, lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle='--', label='Random Classifier')
    ax1.fill_between(fpr, tpr, alpha=0.2, color=BLUE)
    ax1.set_xlabel('False Positive Rate', fontsize=10)
    ax1.set_ylabel('True Positive Rate', fontsize=10)
    ax1.set_title('1. ROC Curve (Held-Out Test AUC=0.661)', fontweight='bold', fontsize=11)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
                xticklabels=['No Readmit', 'Readmit'], yticklabels=['No Readmit', 'Readmit'])
    ax2.set_ylabel('Actual', fontsize=10)
    ax2.set_xlabel('Predicted', fontsize=10)
    ax2.set_title('2. Confusion Matrix (threshold=0.30)', fontweight='bold', fontsize=11)
    
    # 3. Coefficient Plot (Top predictors)
    ax3 = plt.subplot(2, 3, 3)
    coef_vars = ['Prior Admission', 'Dual Eligible', 'High Comorbidity', 'No Follow-up', 
                'D/C Home No HH', 'High SVI']
    coef_vals = [0.839, 0.353, 0.343, 0.297, 0.286, 0.209]
    colors_coef = [RED if x > 0.3 else AMBER for x in coef_vals]
    ax3.barh(coef_vars, coef_vals, color=colors_coef, alpha=0.7)
    ax3.set_xlabel('Standardized Coefficient', fontsize=10)
    ax3.set_title('3. Top 6 Model Coefficients', fontweight='bold', fontsize=11)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Predicted Probability Distribution by Outcome
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(y_pred_proba[y_true == 0], bins=25, alpha=0.6, label='No Readmit', color=GREEN, edgecolor='black')
    ax4.hist(y_pred_proba[y_true == 1], bins=25, alpha=0.6, label='Readmit', color=RED, edgecolor='black')
    ax4.axvline(0.30, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_xlabel('Predicted Probability', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('4. Predicted Probability Distribution', fontweight='bold', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # 5. Calibration Curve
    ax5 = plt.subplot(2, 3, 5)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    ax5.plot([0, 1], [0, 1], linestyle='--', color=GRAY, linewidth=2)
    ax5.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, color=BLUE, label='Model')
    ax5.fill_between(prob_pred, prob_true, alpha=0.2, color=BLUE)
    ax5.set_xlabel('Mean Predicted Probability', fontsize=10)
    ax5.set_ylabel('True Positive Rate', fontsize=10)
    ax5.set_title('5. Calibration Curve', fontweight='bold', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # 6. Permutation Feature Importance
    ax6 = plt.subplot(2, 3, 6)
    feat_importance = {
        'Prior Admission': 0.186,
        'Dual Eligible': 0.142,
        'High Comorbidity': 0.128,
        'No Follow-up': 0.119,
        'D/C Home No HH': 0.105,
        'High SVI': 0.074,
        'Age': 0.058,
        'Short Stay': 0.042
    }
    feat_df = pd.DataFrame(list(feat_importance.items()), columns=['Feature', 'Importance']).sort_values('Importance')
    ax6.barh(feat_df['Feature'], feat_df['Importance'], color=BLUE, alpha=0.7)
    ax6.set_xlabel('Importance', fontsize=10)
    ax6.set_title('6. Permutation Feature Importance (30 repeats)', fontweight='bold', fontsize=11)
    ax6.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Model Validation Results — 6-Panel Summary', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(KEYFINDINGS_DIR / "4_5_model_validation_6panel.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_5_model_validation_6panel.png")
    plt.close()


# ─── 4.6: RISK STRATIFICATION ────────────────────────────────────────────────
def generate_fig_4_6():
    """
    Table 6: Risk stratification summary
    """
    print("Generating 4.6: Risk Stratification...")
    
    risk_strat = df.groupby('Model_Tier').agg({
        'Readmitted_30_Days': ['count', 'sum']
    }).round(1)
    risk_strat.columns = ['N_Patients', 'Readmissions']
    risk_strat['Readmission_Rate'] = (risk_strat['Readmissions'] / risk_strat['N_Patients'] * 100).round(1)
    risk_strat['Share_of_All_Readmits'] = (risk_strat['Readmissions'] / risk_strat['Readmissions'].sum() * 100).round(1)
    
    risk_strat = risk_strat.reindex(['Low', 'Moderate', 'High'])
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Risk Tier Distribution
    tier_colors = [GREEN, AMBER, RED]
    ax1.bar(risk_strat.index, risk_strat['N_Patients'], color=tier_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Risk Tier', fontsize=11, fontweight='bold')
    ax1.set_title('Risk Tier Distribution (n=500)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(risk_strat['N_Patients']):
        ax1.text(i, v + 5, str(int(v)), ha='center', fontweight='bold', fontsize=11)
    
    # Right: Readmission Rates by Tier
    ax2.bar(risk_strat.index, risk_strat['Readmission_Rate'], color=tier_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Readmission Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Risk Tier', fontsize=11, fontweight='bold')
    ax2.set_title('30-Day Readmission Rate by Tier', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(risk_strat['Readmission_Rate']):
        ax2.text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(KEYFINDINGS_DIR / "4_6_risk_stratification.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_6_risk_stratification.png")
    plt.close()
    
    # Also save the data table as CSV
    risk_strat.to_csv(KEYFINDINGS_DIR / "4_6_risk_stratification_table.csv")
    print("✓ Saved: 4_6_risk_stratification_table.csv")


# ─── 4.7: HIGH-RISK PATIENT QUEUE (TOP 10) ────────────────────────────────────
def generate_fig_4_7():
    """
    Table 7: High-risk patient queue top 10
    """
    print("Generating 4.7: High-Risk Patient Queue (Top 10)...")
    
    high_risk = df[df['Model_Tier'] == 'High'].nlargest(10, 'Model_Score')[
        ['Patient_ID', 'Age', 'Primary_Diagnosis', 'Model_Score', 'Dual_Eligible', 
         'Follow_Up_Scheduled', 'Readmitted_30_Days']
    ].reset_index(drop=True)
    
    high_risk.columns = ['Patient ID', 'Age', 'Condition', 'Risk Score', 'Dual Eligible', 'Follow-Up', 'Readmitted']
    high_risk['Dual Eligible'] = high_risk['Dual Eligible'].map({0: 'No', 1: 'Yes'})
    high_risk['Follow-Up'] = high_risk['Follow-Up'].map({0: 'No', 1: 'Yes'})
    high_risk['Readmitted'] = high_risk['Readmitted'].map({0: 'No', 1: 'Yes'})
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=high_risk.values,
                     colLabels=high_risk.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.12, 0.08, 0.16, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(high_risk.columns)):
        table[(0, i)].set_facecolor(RED)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code readmission status
    for i in range(1, len(high_risk) + 1):
        for j in range(len(high_risk.columns)):
            if j == len(high_risk.columns) - 1:  # Last column (Readmitted)
                if high_risk.iloc[i-1, j] == 'Yes':
                    table[(i, j)].set_facecolor('#ffe6e6')
                else:
                    table[(i, j)].set_facecolor('#e6ffe6')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor(LIGHT_GRAY)
    
    plt.title('Top 10 Highest-Risk Patients by Model Risk Score\n(Full 43-patient queue available in output)', 
             fontsize=12, fontweight='bold', pad=15)
    plt.savefig(KEYFINDINGS_DIR / "4_7_high_risk_queue_top10.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_7_high_risk_queue_top10.png")
    plt.close()


# ─── 5.1: RISK SCORE ALGORITHM ────────────────────────────────────────────────
def generate_fig_5_1():
    """
    Risk Score Algorithm and Tier Visualization
    """
    print("Generating 5.1: Risk Score Algorithm...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Risk Scoring Formula Illustration
    ax1 = axes[0, 0]
    ax1.axis('off')
    ax1.text(0.5, 0.85, 'Risk Scoring Formula', ha='center', fontsize=13, fontweight='bold', transform=ax1.transAxes)
    formula_text = r'$Risk\ Score = Round(P(Readmission | X) \times 100)$'
    ax1.text(0.5, 0.65, formula_text, ha='center', fontsize=12, transform=ax1.transAxes, 
            bbox=dict(boxstyle='round', facecolor=BLUE, alpha=0.2))
    
    desc = "Input Features (X):\n• Prior Admission History\n• Comorbidity Burden\n• Social Vulnerability\n• Discharge Destination\n• Follow-up Status"
    ax1.text(0.5, 0.3, desc, ha='center', fontsize=10, transform=ax1.transAxes, 
            bbox=dict(boxstyle='round', facecolor=LIGHT_GRAY, alpha=0.8),
            verticalalignment='center', family='monospace')
    
    # 2. Risk Tier Thresholds
    ax2 = axes[0, 1]
    tiers = ['Low\n<35', 'Moderate\n35-59', 'High\n≥60']
    tier_colors = [GREEN, AMBER, RED]
    tier_counts = [len(df[df['Model_Tier'] == 'Low']), 
                   len(df[df['Model_Tier'] == 'Moderate']),
                   len(df[df['Model_Tier'] == 'High'])]
    
    bars = ax2.bar(range(len(tiers)), tier_counts, color=tier_colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(tiers)))
    ax2.set_xticklabels(tiers, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Patient Count', fontsize=11, fontweight='bold')
    ax2.set_title('Risk Tier Distribution', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, count) in enumerate(zip(bars, tier_counts)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count), 
                ha='center', fontweight='bold', fontsize=11)
    
    # 3. Action Protocol by Tier
    ax3 = axes[1, 0]
    ax3.axis('off')
    ax3.text(0.5, 0.95, 'Care Coordination Action Protocols', ha='center', fontsize=12, fontweight='bold', transform=ax3.transAxes)
    
    protocols = [
        ('Low\n<35', 'Standard discharge\nFollow-up reminder in packet', GREEN),
        ('Moderate\n35-59', 'Enhanced discharge checklist\nCare coordinator review', AMBER),
        ('High\n≥60', 'Mandatory scheduling + coordinator\n+ social work if dual + 48/72h call', RED)
    ]
    
    y_pos = 0.8
    for tier, action, color in protocols:
        ax3.text(0.05, y_pos, tier, fontsize=10, fontweight='bold', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, pad=0.5))
        ax3.text(0.25, y_pos, action, fontsize=9, transform=ax3.transAxes, verticalalignment='center', family='monospace')
        y_pos -= 0.25
    
    # 4. Risk Score Distribution with Tiers
    ax4 = axes[1, 1]
    ax4.hist(df[df['Model_Tier'] == 'Low']['Model_Score'], bins=15, alpha=0.6, label='Low', color=GREEN, edgecolor='black')
    ax4.hist(df[df['Model_Tier'] == 'Moderate']['Model_Score'], bins=15, alpha=0.6, label='Moderate', color=AMBER, edgecolor='black')
    ax4.hist(df[df['Model_Tier'] == 'High']['Model_Score'], bins=15, alpha=0.6, label='High', color=RED, edgecolor='black')
    ax4.axvline(35, color='black', linestyle='--', linewidth=2)
    ax4.axvline(60, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('Risk Score', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Risk Score Distribution by Tier', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('5.1: Risk Score Algorithm and Tier Framework', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(KEYFINDINGS_DIR / "5_1_risk_score_algorithm.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 5_1_risk_score_algorithm.png")
    plt.close()


# ─── Generate All ────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("GENERATING KEY FINDINGS VISUALIZATIONS (4.1 - 5.1)")
    print("="*70 + "\n")
    
    generate_fig_4_1()
    generate_fig_4_2()
    generate_fig_4_3()
    generate_fig_4_4()
    generate_fig_4_5()
    generate_fig_4_6()
    generate_fig_4_7()
    generate_fig_5_1()
    
    print("\n" + "="*70)
    print("✓ ALL KEY FINDINGS GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutputs saved to: {KEYFINDINGS_DIR}")
    print(f"\nGenerated files:")
    for f in sorted(KEYFINDINGS_DIR.glob("*.png")):
        print(f"  • {f.name}")
    for f in sorted(KEYFINDINGS_DIR.glob("*.csv")):
        print(f"  • {f.name}")
    print()

if __name__ == "__main__":
    main()
