import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide', page_title='HRRP Dashboard')

@st.cache_data
def load_data(path='FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv'):
    df = pd.read_csv(path)
    df = df.dropna(subset=['Excess Readmission Ratio']).copy()
    df['Number of Discharges'] = pd.to_numeric(df['Number of Discharges'], errors='coerce')
    return df

measure_map = {
    'READM-30-AMI-HRRP':      'Heart Attack (AMI)',
    'READM-30-HF-HRRP':       'Heart Failure (HF)',
    'READM-30-PN-HRRP':       'Pneumonia (PN)',
    'READM-30-COPD-HRRP':     'COPD',
    'READM-30-HIP-KNEE-HRRP': 'Hip/Knee (THA/TKA)',
    'READM-30-CABG-HRRP':     'Heart Surgery (CABG)',
}

# Load
df = load_data()
# Map common US state abbreviations to full state names for clearer UI
state_map = {
    'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California',
    'CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia',
    'HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa',
    'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland',
    'MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri',
    'MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey',
    'NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio',
    'OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina',
    'SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont',
    'VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming',
    'DC':'District of Columbia','PR':'Puerto Rico'
}
# Normalize and map; if mapping missing, keep original value
df['State'] = df['State'].astype(str).str.upper().map(state_map).fillna(df['State'])

df['Condition'] = df['Measure Name'].map(measure_map)
df['Excess_Flag'] = df['Excess Readmission Ratio'] > 1.0

# Sidebar filters
st.sidebar.title('Filters')
all_states = sorted(df['State'].dropna().unique())
states = st.sidebar.multiselect('State', options=all_states, default=all_states)
all_conditions = sorted(df['Condition'].dropna().unique())
conditions = st.sidebar.multiselect('Condition', options=all_conditions, default=all_conditions)
min_discharges = int(df['Number of Discharges'].min(skipna=True) or 0)
max_discharges = int(df['Number of Discharges'].max(skipna=True) or 0)
discharges = st.sidebar.slider('Min Number of Discharges', min_value=min_discharges, max_value=max_discharges, value=min_discharges)

filtered = df[df['State'].isin(states) & df['Condition'].isin(conditions) & (df['Number of Discharges'] >= discharges)].copy()

# Layout
st.title('HRRP Dashboard — FY 2026')
st.markdown('Interactive dashboard: filter by state, condition, and discharge minimum.')

# Data source and context
st.markdown(
    "**Data source:** [CMS Hospital Readmissions Reduction Program dataset]"
    "(https://data.cms.gov/provider-data/dataset/9n3s-kdb3)  \n"
    "**Released:** February 25, 2026  \n"
    "**Last Modified:** January 26, 2026  \n\n"
    "In October 2012, CMS began reducing Medicare payments for subsection(d) hospitals with excess readmissions under the Hospital Readmissions Reduction Program (HRRP). "
    "Excess readmissions are measured by a ratio, calculated by dividing a hospital's predicted rate of readmissions for heart attack (AMI), heart failure (HF), pneumonia, chronic obstructive pulmonary disease (COPD), hip/knee replacement (THA/TKA), and coronary artery bypass graft surgery (CABG) by the expected rate of readmissions, based on an average hospital with similar patients."
)

# Key metrics
col1, col2, col3 = st.columns(3)
col1.metric('Records', f"{len(filtered):,}")
col2.metric('Unique Hospitals', f"{filtered['Facility ID'].nunique():,}")
col3.metric('%% with ERR > 1', f"{filtered['Excess_Flag'].mean() * 100:.1f}%")

# Figure row
st.subheader('Distributions & Shares')
fig1, ax1 = plt.subplots(figsize=(9, 4))
if len(conditions) > 1:
    sns.histplot(filtered, x='Excess Readmission Ratio', hue='Condition', element='step', stat='count', common_norm=False, kde=False, ax=ax1, palette='tab10')
else:
    sns.histplot(filtered['Excess Readmission Ratio'], bins=40, ax=ax1, color='#2E86C1')
ax1.axvline(1.0, color='red', linestyle='--')
ax1.set_xlabel('Excess Readmission Ratio (ERR)')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of ERR')
fig1.tight_layout()
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(7, 4))
excess_rate = filtered.groupby('Condition')['Excess_Flag'].mean().sort_values(ascending=False) * 100
sns.barplot(x=excess_rate.values, y=excess_rate.index, palette='Blues_r', ax=ax2)
ax2.set_xlabel('% Hospitals with ERR > 1')
ax2.set_title('Share of Hospitals Exceeding Benchmark by Condition')
fig2.tight_layout()
st.pyplot(fig2)

# State-level table and download
st.subheader('State Summary')
state_summary = filtered.groupby('State').agg(
    records=('Excess_Flag', 'count'),
    excess=('Excess_Flag', 'sum')
).reset_index()
state_summary['pct_excess'] = state_summary['excess'] / state_summary['records'] * 100
st.dataframe(state_summary.sort_values('pct_excess', ascending=False).reset_index(drop=True))

csv = state_summary.to_csv(index=False).encode('utf-8')
st.download_button('Download State Summary CSV', data=csv, file_name='state_summary.csv', mime='text/csv')

st.caption('Files saved to the `outputs/` folder by the offline notebook; this app displays figures inline.')
