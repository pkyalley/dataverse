# Telco Customer Churn Analysis & Interactive Dashboard

This project analyzes **Telco Customer Churn** data to explore, visualize, and predict customer churn. Through advanced data analysis, visualization, and machine learning, the goal is to provide actionable insights for improving customer retention.

## Key Features:

- **Data Analysis**: Comprehensive **Exploratory Data Analysis (EDA)** to uncover key drivers of customer churn.
- **Predictive Modeling**: Machine learning models to predict the likelihood of customer churn.

## Interactive Dashboards:
- Built with **Streamlit (Python)** for dynamic and quick visualizations.
- Created **Shiny App (R)** for in-depth, interactive data exploration and statistical analysis.

## Tools & Libraries:
- **Python**: 
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Streamlit
- **RStudio**:
  - Shiny
  - ggplot2
  - corrplot
  - dplyr

## What’s Inside:
1. **Data Preprocessing**: Cleaned and transformed the data for analysis.
2. **EDA**: Visualized churn trends by customer characteristics (gender, contract type, etc.).
3. **Machine Learning Models**: Logistic Regression, Random Forest for churn prediction.
4. **Interactive Dashboards**: Real-time filtering options and visualizations.
5. **Downloadable Reports**: Ability to download the filtered data as CSV.

## Business Insights & Recommendations:
1. **Month-to-Month Contracts**: Customers with month-to-month contracts show higher churn, which suggests the need to incentivize long-term plans.
2. **Target High-Risk Groups**: Customers with high monthly charges and short tenure are more likely to churn, suggesting opportunities for personalized retention offers.
3. **Improve Support Services**: Lack of tech support and online services correlates with higher churn rates, emphasizing the need for better support infrastructure.

## Why Python & R?
- **Python (VS Code)** was used for data processing, machine learning modeling, and creating a Streamlit dashboard due to its strong libraries and ease of integration with machine learning tools.
- **RStudio & Shiny** were used for their superior visualization capabilities and customizable, interactive dashboard features, making them ideal for presenting business insights and detailed statistical analysis.

## Running the Project:
1. **Clone this repository**.
2. Install necessary libraries:
   - For Python: `pip install -r requirements.txt`
   - For R: Install dependencies in RStudio
3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
