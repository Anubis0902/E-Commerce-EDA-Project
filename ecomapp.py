import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

st.set_page_config(page_title="E-Commerce Data Analysis", layout="wide")

# ---- Load Data ----
@st.cache_data
def load_data():
    data = pd.read_csv("D:\\ATHARV\\CODING\\Ecommerce_Delivery_Analytics_New.csv")
    data['Polarity'] = data['Customer Feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    data['Review Type'] = data['Polarity'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
    data['Delivery Status'] = data['Delivery Time (Minutes)'].apply(
        lambda m: 'Within 15 Minutes' if m <= 15 else 'Within 30 Minutes' if m <= 30 else 'More than 30 Minutes'
    )
    data['Order Value Category'] = data['Order Value (INR)'].apply(
        lambda v: 'low' if v < 500 else 'medium' if v < 1000 else 'high'
    )
    return data

data = load_data()

# ---- Sidebar Navigation ----
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio(
    "Go to Section",
    [
        "📌 Overview",
        "💬 Sentiment Analysis",
        "📊 Univariate Analysis",
        "🔗 Bivariate Analysis",
        "⚠️ Data Issues"
    ]
)

# ---- 1. Overview ----
if section == "📌 Overview":
    st.title("📦 E-Commerce Dataset Overview")

    st.markdown("""
    This dataset provides insights into grocery delivery services like **Blinkit**, **Swiggy Instamart**, and **JioMart**.
    It helps analyze customer satisfaction, delivery performance, and service efficiency.
    """)

    st.subheader("🧩 Dataset Information")
    st.write("Shape:", data.shape)
    st.write("Missing Values:")
    st.dataframe(data.isnull().sum())
    st.write("Duplicates:", data.duplicated().sum())

    st.subheader("📊 Summary Statistics")
    st.dataframe(data.describe())

    st.subheader("📋 Column Categories")
    st.markdown("""
    - **Numerical:** Delivery Time (Minutes), Order Value (INR)  
    - **Categorical:** Service Rating, Delivery Delay, Refund Requested, Product Category, Platform  
    - **Text:** Customer Feedback, Order Date & Time
    """)

# ---- 2. Sentiment Analysis ----
elif section == "💬 Sentiment Analysis":
    st.title("💬 Customer Sentiment Analysis")

    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    data['Review Type'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
    ax.set_title("Sentiment Distribution (Positive, Negative, Neutral)")
    st.pyplot(fig)

    st.subheader("Polarity Distribution")
    fig, ax = plt.subplots()
    sns.kdeplot(data['Polarity'], fill=True, ax=ax)
    st.pyplot(fig)

    st.write("Sample Feedbacks:")
    st.dataframe(data[['Customer Feedback', 'Review Type']].head(10))

# ---- 3. Univariate Analysis ----
elif section == "📊 Univariate Analysis":
    st.title("📊 Univariate Analysis")

    st.subheader("Delivery Time Distribution")
    fig, ax = plt.subplots()
    sns.kdeplot(data['Delivery Time (Minutes)'], fill=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Order Value Distribution")
    fig, ax = plt.subplots()
    sns.kdeplot(data['Order Value (INR)'], fill=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Service Rating Distribution")
    fig, ax = plt.subplots()
    sns.kdeplot(data['Service Rating'], fill=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Delivery Status Percentages")
    st.dataframe((data['Delivery Status'].value_counts(normalize=True) * 100).round(2))

    st.subheader("Refund Requests (%)")
    st.dataframe((data['Refund Requested'].value_counts(normalize=True) * 100).round(2))

    st.subheader("Product Category Distribution")
    fig, ax = plt.subplots()
    data['Product Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

    st.subheader("Platform Distribution")
    fig, ax = plt.subplots()
    data['Platform'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

# ---- 4. Bivariate Analysis ----
elif section == "🔗 Bivariate Analysis":
    st.title("🔗 Bivariate Analysis")

    st.subheader("Order Value Category vs Delivery Status")
    fig, ax = plt.subplots()
    pd.crosstab(data['Order Value Category'], data['Delivery Status']).plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Delivery Status vs Service Rating")
    fig, ax = plt.subplots()
    pd.crosstab(data['Delivery Status'], data['Service Rating']).plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Refund Requests by Product Category")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(data['Product Category'], data['Refund Requested']), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("Orders by Product Category on Platform")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(data['Product Category'], data['Platform']), annot=True, fmt='d', cmap='Reds', ax=ax)
    st.pyplot(fig)

# ---- 5. Data Issues ----
elif section == "⚠️ Data Issues":
    st.title("⚠️ Data Quality Check")

    st.write("Issue Found: Some 'Personal Care' items marked as 'Not fresh, disappointed' — logically incorrect.")

    invalid = (data['Customer Feedback'].str.contains('Not fresh, disappointed.', na=False)) & \
              (data['Product Category'].str.contains('Personal Care', na=False))
    invalid_rows = data.loc[invalid]

    if invalid_rows.empty:
        st.success("✅ No invalid feedback found.")
    else:
        st.error(f"⚠️ Found {len(invalid_rows)} invalid feedback entries.")
        st.dataframe(invalid_rows)
