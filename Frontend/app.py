import os
from pathlib import Path
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DFIO | Smart Inventory",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports after set_page_config to avoid Streamlit warning
from api import (
    API_BASE_URL,
    AGENT_FORECAST_ENDPOINT,
    APPROVAL_ENDPOINT,
    API_HEADERS,
    fetch_model_status,
    fetch_forecast
)
from components.sidebar import render_sidebar
from components.inputs import render_inputs
from components.results import render_results
from components.approval import render_approval
import base64

st.markdown("""
<style>
[data-testid="stVerticalBlockBorderWrapper"] {
    margin-top: 0px !important;
    margin-bottom: 4px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD CSS + BACKGROUND IMAGE
# =========================================================

FRONTEND_DIR = Path(__file__).resolve().parent

with open(FRONTEND_DIR / "styles.css", "r", encoding="utf-8") as f:
    css = f.read()

try:
    with open(FRONTEND_DIR / "images" / "Store.jpg", "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    css = css.replace(
        "BACKGROUND_IMAGE_HERE",
        f"data:image/jpeg;base64,{encoded}"
    )

except FileNotFoundError:
    css = css.replace(
        "BACKGROUND_IMAGE_HERE",
        ""
    )

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)

if "forecast_response" not in st.session_state:
    st.session_state.forecast_response = None
if "approval_response" not in st.session_state:
    st.session_state.approval_response = None


model_status = fetch_model_status()

# =========================================================
# SIDEBAR
# =========================================================

region, category, time_range = render_sidebar()



# =========================================================
# MAIN TITLE / WELCOME AREA
# =========================================================
st.markdown("""
<style>
h1 {
    margin-bottom: 5px !important;
}

h2 {
    margin-top: 0px !important;
    margin-bottom: 5px !important;
}

[data-testid="stCaptionContainer"] {
    margin-top: 0px !important;
    margin-bottom: -15px !important;
}

hr {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📦 AI Inventory Assistant")

st.subheader(
    "Demand Forecasting & Inventory Optimization for Local Retailers"
)

st.markdown(
    """
    <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6; margin-top: 6px; margin-bottom: 18px;">
        An intelligent decision-support platform designed to help retail managers accurately forecast SKU-level demand across diverse store locations and categories. By analyzing historical sales trends, promotional campaigns, pricing dynamics, and local market conditions, the system detects stockout risks and overstock positions in real time. It calculates optimal replenishment quantities with safety buffers and delivers actionable AI-guided recommendations to streamline inventory management and prevent lost sales.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()
# =========================================================
# AI ENGINE STATUS
# =========================================================
status1, status2, status3 = st.columns(3)

with status1:
    with st.container(border=True):
        st.metric(
            "🧠 Forecast Engine",
            "READY" if model_status and model_status.get("status") == "online" else "OFFLINE"
        )
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
with status2:
    with st.container(border=True):
        st.metric(
            "📊 Model Accuracy",
            f"{model_status['accuracy'] * 100:.1f}%"
            if model_status and isinstance(model_status.get("accuracy"), (int, float))
            else "N/A"
        )

with status3:
    with st.container(border=True):
        st.metric(
            "⚡ Prediction Mode",
            "REAL-TIME"
        )
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# =========================================================
# INPUTS
# =========================================================

inputs = render_inputs()

# =========================================================
# FORECAST BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

    test = st.button(
        "⚡ RUN AI DEMAND FORECAST",
        use_container_width=True
    )


# =========================================================
# RESULT
# =========================================================

if test:
    missing_fields = []
    required_values = {
        "Region": region,
        "Category": category,
        "Forecast Period": time_range,
        "Store Name": inputs.get("store_id"),
        "Product ID": inputs.get("product_id"),
        "Weather Condition": inputs.get("weather_condition"),
        "Epidemic": inputs.get("epidemic"),
    }
    for field_name, value in required_values.items():
        if value is None:
            missing_fields.append(field_name)

    if missing_fields:
        st.error("⚠️ Please select all required fields before running the forecast.")
        st.warning("Missing: " + ", ".join(missing_fields))
    else:
        payload = {
            "date": "2024-02-15",
            "store_id": f"S{int(inputs.get('store_id').split()[-1]):03d}",
            "product_id": f"P{int(inputs.get('product_id')[1:]):04d}",
            "category": category,
            "region": region,
            "inventory_level": int(inputs.get("inventory_level")),
            "units_sold": int(inputs.get("units_sold")),
            "units_ordered": int(inputs.get("units_ordered")),
            "price": float(inputs.get("price")),
            "discount": float(inputs.get("discount_rate")),
            "competitor_pricing": float(inputs.get("competitor_pricing")),
            "weather_condition": inputs.get("weather_condition"),
            "promotion": int(inputs.get("promotion")),
            "seasonality": "Spring",
            "epidemic": int(inputs.get("epidemic")),
            "forecast_period": time_range,
        }

        try:
            with st.spinner("🧠 AI model is generating the forecast..."):
                response = fetch_forecast(payload)

            if response.status_code == 200:
                st.session_state.forecast_response = response.json()
                st.session_state.approval_response = None
                st.success("✅ AI Demand Forecast Completed")
            elif response.status_code in (400, 422):
                try:
                    error_data = response.json()
                    error = error_data.get("error", {})
                    st.error("❌ " + error.get("message", "Invalid input values."))
                    for field, message in error.get("fields", {}).items():
                        st.warning(f"{field}: {message}")
                except ValueError:
                    st.error("❌ Backend validation failed.")
            elif response.status_code == 404:
                st.error("❌ Store or product was not found.")
            elif response.status_code == 409:
                st.error("⚠️ Inventory data is stale or conflicting.")
            elif response.status_code == 429:
                st.error("⚠️ Too many requests. Please try again later.")
            elif response.status_code in (500, 503):
                st.error("⚠️ Forecast service is currently unavailable.")
            else:
                st.error(f"❌ Backend returned HTTP {response.status_code}")
        except Exception as e:
            st.error(f"❌ Network or connection error: {e}")

render_results(model_status)
render_approval(API_HEADERS, APPROVAL_ENDPOINT)
