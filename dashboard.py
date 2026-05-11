import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI House Price Dashboard",
    layout="wide"
)

# ==========================================
# ADVANCED MODERN UI
# ==========================================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #f3f4f6;
}

/* TITLE */
h1 {
    color: #0f172a;
    font-size: 45px;
    font-weight: bold;
}

/* METRIC CARDS */
[data-testid="metric-container"] {
    background: white;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    border-left: 8px solid #4f46e5;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1d4ed8,
        #6d28d9
    );
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(
        to right,
        #2563eb,
        #7c3aed
    );

    color: white;
    border-radius: 12px;
    border: none;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

/* TABLE */
thead tr th {
    background-color: #7c3aed !important;
    color: white !important;
}

/* SLIDERS */
.stSlider label {
    color: #1e293b !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv(
    r"C:\Users\maste\Downloads\house_price_regression_dataset.csv"
)

# ==========================================
# LOAD MODELS
# ==========================================
linear_model = joblib.load("linear_model.pkl")

rf_model = joblib.load("random_forest_model.pkl")

xgb_model = joblib.load("xgboost_model.pkl")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🏠 AI House Price")

st.sidebar.markdown("""
### Prediction System

✅ Dashboard  
✅ Real-Time Prediction  
✅ Recommendations  
✅ Data Analysis  
✅ Model Comparison  
""")

st.sidebar.markdown("---")

st.sidebar.subheader("ABOUT DATASET")

st.sidebar.write("""
This dataset contains house
information and prices to
predict estimated prices
using Machine Learning.
""")

st.sidebar.metric(
    "TOTAL RECORDS",
    len(df)
)

st.sidebar.metric(
    "TOTAL FEATURES",
    7
)

# ==========================================
# TITLE
# ==========================================
st.title("🏠 AI House Price Prediction Dashboard")

st.write(
    "Predict house prices using Multiple Machine Learning Models"
)

# ==========================================
# DASHBOARD OVERVIEW
# ==========================================
st.subheader("📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Houses",
    len(df)
)

col2.metric(
    "Average Price",
    f"₹ {int(df['House_Price'].mean())}"
)

col3.metric(
    "Highest Price",
    f"₹ {int(df['House_Price'].max())}"
)

col4.metric(
    "Lowest Price",
    f"₹ {int(df['House_Price'].min())}"
)

# ==========================================
# VISUALIZATION
# ==========================================
fig1 = px.scatter(
    df,
    x="Square_Footage",
    y="House_Price",
    color="Num_Bedrooms",
    title="Square Footage vs House Price"
)

# ==========================================
# MODEL ACCURACY
# ==========================================
comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        99.84,
        99.40,
        99.52
    ]
})

fig2 = px.bar(
    comparison,
    x="Model",
    y="Accuracy",
    title="Model Accuracy Comparison"
)

# ==========================================
# CHARTS SIDE BY SIDE
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================================
# REAL-TIME PREDICTION
# ==========================================
st.subheader("🤖 Real-Time Prediction Inputs")

square_footage = st.slider(
    "Square Footage",
    500,
    10000,
    2500
)

num_bedrooms = st.slider(
    "Number of Bedrooms",
    1,
    10,
    3
)

num_bathrooms = st.slider(
    "Number of Bathrooms",
    1,
    10,
    2
)

year_built = st.slider(
    "Year Built",
    1950,
    2025,
    2015
)

lot_size = st.slider(
    "Lot Size",
    1,
    10,
    2
)

garage_size = st.slider(
    "Garage Size",
    0,
    5,
    1
)

neighborhood_quality = st.slider(
    "Neighborhood Quality",
    1,
    10,
    6
)

# ==========================================
# INPUT DATA
# ==========================================
input_data = [[
    square_footage,
    num_bedrooms,
    num_bathrooms,
    year_built,
    lot_size,
    garage_size,
    neighborhood_quality
]]

# ==========================================
# PREDICTIONS
# ==========================================
linear_prediction = linear_model.predict(
    input_data
)[0]

rf_prediction = rf_model.predict(
    input_data
)[0]

xgb_prediction = xgb_model.predict(
    input_data
)[0]

# ==========================================
# PREDICTION BOXES
# ==========================================
st.subheader("📈 Real-Time Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        f"🏠 Linear Regression Prediction\n\n₹ {int(linear_prediction)}"
    )

with col2:
    st.info(
        f"🌳 Random Forest Prediction\n\n₹ {int(rf_prediction)}"
    )

with col3:
    st.error(
        f"⚡ XGBoost Prediction\n\n₹ {int(xgb_prediction)}"
    )

# ==========================================
# RECOMMENDATION SYSTEM
# ==========================================
st.subheader("🏘 Recommended Similar Houses")

recommended = df[
    (df["Num_Bedrooms"] == num_bedrooms)
    &
    (df["Num_Bathrooms"] == num_bathrooms)
]

st.dataframe(
    recommended[[
        "Square_Footage",
        "Num_Bedrooms",
        "Num_Bathrooms",
        "House_Price"
    ]].head(5)
)

# ==========================================
# AI INSIGHTS
# ==========================================
st.subheader("🧠 AI Insights")

avg_price = df["House_Price"].mean()

if rf_prediction > avg_price:

    st.success(
        "This house is considered a premium property."
    )

else:

    st.info(
        "This house is considered budget-friendly."
    )

# ==========================================
# TOP EXPENSIVE HOUSES
# ==========================================
st.subheader("🏆 Top 5 Most Expensive Houses")

top_houses = df.sort_values(
    by="House_Price",
    ascending=False
)

st.dataframe(
    top_houses.head(5)
)

# ==========================================
# FOOTER
# ==========================================
st.write(
    "✅ Built using Streamlit | Machine Learning Project"
)