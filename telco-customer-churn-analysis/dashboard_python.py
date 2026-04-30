import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set up Streamlit page
st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')  # Updated file path
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df

df = load_data()

# Title
st.title('📊 Telco Customer Churn Dashboard')
st.markdown('### Explore Churn Trends and Insights')

# Explanation of Churn
st.markdown("""
### Understanding Churn
- **Churn: No** - These are customers who have not discontinued their service with the company.
- **Churn: Yes** - These are customers who have discontinued their service with the company.
""")

# Show dataset preview
if st.checkbox('Show raw data'):
    st.dataframe(df)

# Filters
st.sidebar.header('Filter Options')
gender = st.sidebar.multiselect('Gender', options=df['gender'].unique(), default=df['gender'].unique())
senior_citizen = st.sidebar.multiselect('Senior Citizen', options=df['SeniorCitizen'].unique(), default=df['SeniorCitizen'].unique())
partner = st.sidebar.multiselect('Partner', options=df['Partner'].unique(), default=df['Partner'].unique())
contract = st.sidebar.multiselect('Contract', options=df['Contract'].unique(), default=df['Contract'].unique())

filtered_df = df[(df['gender'].isin(gender)) & 
                 (df['SeniorCitizen'].isin(senior_citizen)) & 
                 (df['Partner'].isin(partner)) & 
                 (df['Contract'].isin(contract))]

# KPIs
total_customers = filtered_df.shape[0]
churned_customers = filtered_df[filtered_df['Churn'] == 'Yes'].shape[0]
churn_rate = churned_customers / total_customers * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned_customers)
col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")

# Plot Selection
st.sidebar.header('Plot Selection')
plots = st.sidebar.multiselect('Select Plots', options=['Churn Distribution', 'Tenure by Churn', 'Contract Type by Churn', 'Monthly Charges by Churn', 'Customer Demographics', 'Service Usage', 'Correlation Heatmap', 'Customer Segmentation'], default=['Churn Distribution', 'Tenure by Churn', 'Contract Type by Churn', 'Monthly Charges by Churn'])

# Churn Distribution Plot
if 'Churn Distribution' in plots:
    st.subheader('Churn Distribution')
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x='Churn', palette='pastel', ax=ax1)
    st.pyplot(fig1)

# Tenure by Churn Plot
if 'Tenure by Churn' in plots:
    st.subheader('Tenure Distribution by Churn')
    fig2, ax2 = plt.subplots()
    sns.histplot(data=filtered_df, x='tenure', hue='Churn', multiple='stack', palette='pastel', ax=ax2)
    st.pyplot(fig2)

# Contract Type by Churn Plot
if 'Contract Type by Churn' in plots:
    st.subheader('Churn by Contract Type')
    fig3, ax3 = plt.subplots()
    sns.countplot(data=filtered_df, x='Contract', hue='Churn', palette='pastel', ax=ax3)
    st.pyplot(fig3)

# Monthly Charges by Churn
if 'Monthly Charges by Churn' in plots:
    st.subheader('Monthly Charges Distribution by Churn')
    fig4, ax4 = plt.subplots()
    sns.histplot(data=filtered_df, x='MonthlyCharges', hue='Churn', multiple='stack', palette='pastel', ax=ax4)
    st.pyplot(fig4)

# Customer Demographics Analysis
if 'Customer Demographics' in plots:
    st.subheader('Churn by Customer Demographics')
    fig5, ax5 = plt.subplots()
    sns.countplot(data=filtered_df, x='gender', hue='Churn', palette='pastel', ax=ax5)
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots()
    sns.countplot(data=filtered_df, x='SeniorCitizen', hue='Churn', palette='pastel', ax=ax6)
    st.pyplot(fig6)

    fig7, ax7 = plt.subplots()
    sns.countplot(data=filtered_df, x='Partner', hue='Churn', palette='pastel', ax=ax7)
    st.pyplot(fig7)

# Service Usage Analysis
if 'Service Usage' in plots:
    st.subheader('Churn by Service Usage')
    fig8, ax8 = plt.subplots()
    sns.countplot(data=filtered_df, x='PhoneService', hue='Churn', palette='pastel', ax=ax8)
    st.pyplot(fig8)

    fig9, ax9 = plt.subplots()
    sns.countplot(data=filtered_df, x='MultipleLines', hue='Churn', palette='pastel', ax=ax9)
    st.pyplot(fig9)

    fig10, ax10 = plt.subplots()
    sns.countplot(data=filtered_df, x='InternetService', hue='Churn', palette='pastel', ax=ax10)
    st.pyplot(fig10)

# Correlation Heatmap
if 'Correlation Heatmap' in plots:
    st.subheader('Correlation Heatmap')
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    fig11, ax11 = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax11)
    st.pyplot(fig11)

# Customer Segmentation
if 'Customer Segmentation' in plots:
    st.subheader('Customer Segmentation')
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters)
    filtered_df['Cluster'] = kmeans.fit_predict(filtered_df[['tenure', 'MonthlyCharges', 'TotalCharges']].dropna())

    # Assign meaningful names to clusters
    cluster_names = {i: f'Cluster {i}' for i in range(n_clusters)}
    filtered_df['ClusterName'] = filtered_df['Cluster'].map(cluster_names)

    fig12, ax12 = plt.subplots()
    sns.scatterplot(data=filtered_df, x='tenure', y='MonthlyCharges', hue='ClusterName', palette='pastel', ax=ax12)
    st.pyplot(fig12)

# Filter and subgroup analysis (optional, interactive)
st.subheader('Filter Analysis by Internet Service Type')
internet_service = st.selectbox('Select Internet Service:', df['InternetService'].unique())
filtered_df_service = df[df['InternetService'] == internet_service]

st.write(f"Total Customers with {internet_service}: {filtered_df_service.shape[0]}")

fig13, ax13 = plt.subplots()
sns.countplot(data=filtered_df_service, x='Churn', palette='pastel', ax=ax13)
st.pyplot(fig13)

# Churn Prediction Model
st.subheader('Churn Prediction Model')

# Prepare data for model training
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")

# User input for prediction
st.sidebar.header('Predict Churn for a New Customer')
tenure = st.sidebar.number_input('Tenure', min_value=0, max_value=100, value=1)
monthly_charges = st.sidebar.number_input('Monthly Charges', min_value=0.0, max_value=1000.0, value=50.0)
total_charges = st.sidebar.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=100.0)

# Predict churn for new customer
new_customer = pd.DataFrame({'tenure': [tenure], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]})
new_customer_scaled = scaler.transform(new_customer)
churn_prediction = model.predict(new_customer_scaled)[0]
churn_probability = model.predict_proba(new_customer_scaled)[0][1]

st.sidebar.write(f"Churn Prediction: {'Yes' if churn_prediction == 1 else 'No'}")
st.sidebar.write(f"Churn Probability: {churn_probability:.2f}")

st.markdown("#### Key Insights")
st.markdown("""
- Customers on **month-to-month** contracts churn the most.
- **Higher monthly charges** often lead to churn.
- Customers with **longer tenure** tend to stay longer.
- **Senior citizens** have a higher churn rate.
- **Customers without partners** have a higher churn rate.
- **Phone service** and **multiple lines** usage impact churn.
""")