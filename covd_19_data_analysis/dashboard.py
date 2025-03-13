import streamlit as st
import pandas as pd
import plotly.express as px

# Set the page config
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

# Title of the Dashboard
st.title("🌍 COVID-19 Interactive Dashboard")
st.markdown("Track global COVID-19 data over time. 📈")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_clean_complete.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter by:")
countries = df['Country/Region'].unique()
selected_country = st.sidebar.selectbox("Select a Country", countries)

# Filter data for selected country
country_data = df[df['Country/Region'] == selected_country]

# KPIs
st.subheader(f"COVID-19 Statistics for {selected_country}")

latest_date = country_data['Date'].max()
latest_data = country_data[country_data['Date'] == latest_date]

confirmed = int(latest_data['Confirmed'].sum())
deaths = int(latest_data['Deaths'].sum())
recovered = int(latest_data['Recovered'].sum())

col1, col2, col3 = st.columns(3)
col1.metric("Confirmed Cases", f"{confirmed:,}")
col2.metric("Deaths", f"{deaths:,}")
col3.metric("Recovered", f"{recovered:,}")

# Plotly Line Chart for Cases Over Time
fig = px.line(country_data, x='Date', y=['Confirmed', 'Deaths', 'Recovered'],
              labels={'value': 'Cases', 'variable': 'Case Type'},
              title=f"COVID-19 Trends Over Time in {selected_country}")

st.plotly_chart(fig, use_container_width=True)

# Raw Data (optional)
with st.expander("📊 Show Raw Data"):
    st.write(country_data)

st.subheader("🌎 Global COVID-19 Cases Map")

# Get latest data by country (we group by country on the latest date)
latest_date = df['Date'].max()
global_latest = df[df['Date'] == latest_date]

fig_map = px.scatter_geo(global_latest,
                         locations="Country/Region",
                         locationmode='country names',
                         color="Confirmed",
                         size="Confirmed",
                         hover_name="Country/Region",
                         projection="natural earth",
                         title=f"Global Spread on {latest_date.date()}",
                         size_max=60)

st.plotly_chart(fig_map, use_container_width=True)

st.subheader(f"📅 Daily New Cases in {selected_country}")

# Calculate daily new cases
country_data_sorted = country_data.sort_values('Date')
country_data_sorted['Daily Confirmed'] = country_data_sorted['Confirmed'].diff().fillna(0)
country_data_sorted['Daily Deaths'] = country_data_sorted['Deaths'].diff().fillna(0)
country_data_sorted['Daily Recovered'] = country_data_sorted['Recovered'].diff().fillna(0)

# Plot Daily New Cases
fig_daily = px.bar(country_data_sorted,
                   x='Date',
                   y='Daily Confirmed',
                   title=f'Daily New Confirmed Cases in {selected_country}',
                   labels={'Daily Confirmed': 'New Cases'})

st.plotly_chart(fig_daily, use_container_width=True)

# Date Range Filter
min_date = df['Date'].min()
max_date = df['Date'].max()

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Filter country data by date range
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
country_data = country_data[(country_data['Date'] >= start_date) & (country_data['Date'] <= end_date)]
