import os
from urllib import response
import requests as http_requests
from django.contrib.sites import requests
import streamlit as st
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
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DFIO | Smart Inventory",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD CSS + BACKGROUND IMAGE
# =========================================================

with open("styles.css", "r") as f:
    css = f.read()

try:
    with open("images/Store.jpg", "rb") as image:
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


API_BASE_URL = os.getenv(
    "DFIO_API_BASE_URL",
    "http://localhost:8000"
).rstrip("/")

FORECAST_ENDPOINT = f"{API_BASE_URL}/api/v1/forecasts"

if "forecast_response" not in st.session_state:
    st.session_state.forecast_response = None
# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # Brand
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">📦</div>
            <div>
                <div class="sidebar-title">DFIO</div>
                <div class="sidebar-subtitle">
                    AI INVENTORY INTELLIGENCE
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # System status
    st.markdown(
        """
        <div class="system-status">
            <span class="status-dot"></span>
            AI SYSTEM ONLINE
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="side-heading">CONTROL PANEL</div>',
        unsafe_allow_html=True
    )

    region = st.selectbox(
    "🌍 Select Region *",
    ["North", "South", "East", "West"],
    index=None,
    placeholder="Choose region"
    )

    category = st.selectbox(
    "📦 Select Category *",
    [
        "Electronics",
        "Clothing",
        "Groceries",
        "Toys",
        "Furniture"
    ],
    index=None,
    placeholder="Choose category"
    )
    time_range = st.selectbox(
    "📅 Forecast Period *",
    [
        "1 week",
        "1 month",
        "3 months",
        "6 months",
        "1 year"
    ],
    index=None,
    placeholder="Choose period"
    )

    # =====================================================
    # CURRENT SELECTION
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.sidebar.subheader("📌 Current Selection")

    selected_region = region if region else "Not selected"
    selected_category = category if category else "Not selected"
    selected_period = time_range if time_range else "Not selected"

    st.caption("REGION")
    st.write(f"🌍 {selected_region}")

    st.caption("CATEGORY")
    st.write(f"📦 {selected_category}")

    st.caption("FORECAST PERIOD")
    st.write(f"📅 {selected_period}")


    # =====================================================
    # AI FORECAST ENGINE
    # =====================================================

    model_col1, model_col2 = st.sidebar.columns(2)

latest_response = st.session_state.forecast_response

if latest_response:

    model_data = latest_response.get("model", {})
    prediction_data = latest_response.get("prediction", {})

    accuracy = model_data.get("accuracy")
    confidence = prediction_data.get("confidence_score")

    accuracy_display = (
        f"{accuracy * 100:.1f}%"
        if isinstance(accuracy, (int, float))
        else "---%"
    )

    confidence_display = (
        f"{confidence * 100:.1f}%"
        if isinstance(confidence, (int, float))
        else "---%"
    )

else:
    accuracy_display = "---%"
    confidence_display = "---%"


with model_col1:
    st.metric(
        "Accuracy",
        accuracy_display
    )


with model_col2:
    st.metric(
        "Confidence",
        confidence_display
    )


    # =====================================================
    # FOOTER
    # =====================================================

    st.sidebar.divider()

    st.sidebar.caption(
        "DFIO • Demand Forecasting & Inventory Optimization"
    )
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

st.caption(
    "AI-powered inventory intelligence for smarter demand prediction, "
    "stock planning and retail decision-making."
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
            "READY"
        )
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
with status2:
    with st.container(border=True):
        st.metric(
            "📊 Model Accuracy",
            "83.5%",
            "+1.5%"
        )

with status3:
    with st.container(border=True):
        st.metric(
            "⚡ Prediction Mode",
            "REAL-TIME"
        )
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION HEADER
# =========================================================

st.subheader("📊 Demand Forecasting Inputs")

st.caption(
    "Enter the current business conditions used by the "
    "AI forecasting engine to predict future demand."
)

# =========================================================
# BASIC INFORMATION CARD
# =========================================================

with st.container(border=True):

    st.markdown("### 🏪 Product & Location Intelligence")

    st.caption(
        "Identify the store, product and geographical market."
    )

    col1, col2, = st.columns(2)

    with col1:
        store_id = st.selectbox(
            "Store Name *",
            ["Store 1", "Store 2", "Store 3", "Store 4"],
            index=None,
            placeholder="Select store"
        )

    with col2:
        product_id = st.selectbox(
            "Product ID *",
            ["P001", "P002", "P003", "P004"],
            index=None,
            placeholder="Select product"
        )

    # with col3:
    #     store_id = st.selectbox(
    #         "Store ID",
    #         ["S001", "S002", "S003", "S004", "S005"],
    #         index=None,
    #         placeholder="Select store"
    #     )
# =========================================================
# MARKET CONDITIONS CARD
# =========================================================

with st.container(border=True):

    st.markdown(
        """
        <div class="input-card-title">
            🌦️ Market & Environmental Conditions
        </div>

        <div class="input-card-description">
            Environmental and pricing factors affecting demand.
        </div>
        """,
        unsafe_allow_html=True
    )

    col4, col5, col6 = st.columns(3)

    with col4:
        weather_condition = st.selectbox(
            "Weather Condition *",
            ["Sunny", "Rainy", "Snowy", "Cloudy"],
            index=None,
            placeholder="Select weather"
        )

    with col5:
        inventory_level = st.number_input(
            "Inventory Level *",
            min_value=0,
            max_value=1000,
            step=1
        )

    with col6:
        discount_rate = st.number_input(
            "Discount Rate (%) *",
            min_value=0.0,
            max_value=100.0,
            step=0.5
        )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# SALES & PROMOTION CARD
# =========================================================

with st.container(border=True):

    st.markdown(
        """
        <div class="input-card-title">
            📈 Sales & Promotion Signals
        </div>

        <div class="input-card-description">
            Historical sales movement and promotional activity.
        </div>
        """,
        unsafe_allow_html=True
    )

    col7, col8, col9 = st.columns(3)

    with col7:
        units_sold = st.number_input(
            "Units Sold *",
            min_value=0,
            max_value=1000,
            step=5
        )

    with col8:
        units_ordered = st.number_input(
            "Units Ordered *",
            min_value=0,
            max_value=1000,
            step=5
        )

    with col9:

        promotion = st.checkbox(
            "Promotion Active *",
            value=False
        )

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# PRODUCT ATTRIBUTE CARD
# =========================================================

with st.container(border=True):

    st.markdown(
        """
        <div class="input-card-title">
            🧬 Product Attribute
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="color:#6f8fb5; font-size:14px; margin-top:6px;">'
        'Additional product characteristics used by the model.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="font-size:16px; margin-top:16px; margin-bottom:10px;">'
        '<b>Epidemic</b> '
        '<span style="color:#6f8fb5; font-size:14px;">'
        '— Indicates an outbreak that can affect product demand.'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )

    epidemic = st.selectbox(
        "Epidemic *",
        [0, 1],
        index=None,
        placeholder="Select value",
        label_visibility="collapsed"
    )

# =========================
# COMPETITOR PRICING
# =========================
with st.container(border=True):

    st.markdown(
        """
        <div class="input-card-title">
            💰 Competitor Pricing
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:#6f8fb5; font-size:14px; margin-top:6px; margin-bottom:10px;">'
        'Competitor product price in ₹.'
        '</div>',
        unsafe_allow_html=True
    )

    competitor_pricing = st.number_input(
        "Competitor Pricing *",
        min_value=0.0,
        max_value=10000.0,
        step=1.0,
        format="%.2f",
        label_visibility="collapsed"
    )

# =========================================================
# LIVE INPUT SUMMARY
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="live-header">
        <span>◉</span>
        CURRENT FORECAST INPUTS
    </div>
    """,
    unsafe_allow_html=True
)

sum1, sum2, sum3, sum4 = st.columns(4)

with sum1:
    st.metric(
        "Inventory",
        inventory_level
    )

with sum2:
    st.metric(
        "Units Sold",
        units_sold
    )

with sum3:
    st.metric(
        "Units Ordered",
        units_ordered
    )

with sum4:
    st.metric(
        "Discount",
        f"{discount_rate:.1f}%"
    )


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

    # =====================================================
    # FRONTEND VALIDATION
    # =====================================================

    missing_fields = []

    required_values = {
        "Region": region,
        "Category": category,
        "Forecast Period": time_range,
        "Store Name": store_id,
        "Product ID": product_id,
        "Weather Condition": weather_condition,
        "Epidemic": epidemic
    }

    for field_name, value in required_values.items():

        if value is None:
            missing_fields.append(field_name)


    # =====================================================
    # VALIDATION ERROR
    # =====================================================

    if missing_fields:

        st.error(
            "⚠️ Please select all required fields before running the forecast."
        )

        st.warning(
            "Missing: " + ", ".join(missing_fields)
        )


    else:

        # =================================================
        # CREATE API PAYLOAD
        # =================================================

        payload = {

            "region": region,

            "category": category,

            "forecast_period": time_range,

            "store_id": store_id,

            "product_id": product_id,

            "weather_condition": weather_condition,

            "inventory_level": int(inventory_level),

            "discount_rate": float(discount_rate),

            "units_sold": int(units_sold),

            "units_ordered": int(units_ordered),

            "promotion_active": bool(promotion),

            "epidemic": int(epidemic),

            "competitor_pricing": float(competitor_pricing)

        }


        # =================================================
        # SEND REQUEST TO BACKEND
        # =================================================

        try:

            with st.spinner(
                "🧠 AI model is generating the forecast..."
            ):

                response = http_requests.post(
                FORECAST_ENDPOINT,
                json=payload,
                timeout=30
                )    


            # =================================================
            # SUCCESS
            # =================================================

            if  response.status_code == 200:

                data = response.json()

                st.session_state.forecast_response = data

                status = data.get(
                    "status",
                    "completed"
                )


                if status == "completed":

                    st.success(
                        "✅ AI Demand Forecast Completed"
                    )


                    prediction = data.get(
                        "prediction",
                        {}
                    )

                    model = data.get(
                        "model",
                        {})


                    # =============================================
                    # GET MODEL OUTPUTS
                    # =============================================

                    predicted_demand = prediction.get(
                        "predicted_demand_units",
                        "N/A"
                    )

                    recommended_order = prediction.get(
                        "recommended_order_quantity",
                        "N/A"
                    )

                    stockout_risk = prediction.get(
                        "expected_stockout_risk"
                    )

                    confidence = prediction.get(
                        "confidence_score"
                    )


                    # =============================================
                    # DISPLAY RESULTS
                    # =============================================

                    col1, col2, col3, col4 = st.columns(4)


                    with col1:

                        st.metric(
                            "📦 Predicted Demand",
                            f"{predicted_demand} units"
                        )


                    with col2:

                        st.metric(
                            "🛒 Recommended Order",
                            f"{recommended_order} units"
                        )


                    with col3:

                        if isinstance(
                            stockout_risk,
                            (int, float)
                        ):

                            risk_display = (
                                f"{stockout_risk * 100:.1f}%"
                            )

                        else:

                            risk_display = "N/A"


                        st.metric(
                            "⚠️ Stockout Risk",
                            risk_display
                        )


                    with col4:

                        if isinstance(
                            confidence,
                            (int, float)
                        ):

                            confidence_display = (
                                f"{confidence * 100:.1f}%"
                            )

                        else:

                            confidence_display = "N/A"


                        st.metric(
                            "🎯 Confidence",
                            confidence_display
                        )


                    # =============================================
                    # MODEL INFORMATION
                    # =============================================

                    accuracy = model.get(
                        "accuracy"
                    )

                    if isinstance(
                        accuracy,
                        (int, float)
                    ):

                        accuracy_display = (
                            f"{accuracy * 100:.1f}%"
                        )

                    else:

                        accuracy_display = "N/A"


                    st.info(

                        f"🤖 Model: "
                        f"{model.get('name', 'N/A')} "

                        f" | Version: "
                        f"{model.get('version', 'N/A')} "

                        f" | Accuracy: "
                        f"{accuracy_display} "

                        f" | Request ID: "
                        f"{data.get('request_id', 'N/A')} "

                        f" | Generated: "
                        f"{data.get('generated_at', 'N/A')}"

                    )


                elif status == "queued":

                    st.info(

                        "⏳ Forecast request has been queued. "

                        f"Request ID: "
                        f"{data.get('request_id', 'N/A')}"

                    )


                else:

                    st.error(
                        "❌ Forecast failed. Please try again."
                    )


            # =================================================
            # VALIDATION ERROR
            # =================================================

            elif response.status_code in (400, 422):

                try:

                    error_data = response.json()

                    error = error_data.get(
                        "error",
                        {}
                    )

                    st.error(

                        "❌ " +
                        error.get(
                            "message",
                            "Invalid input values."
                        )

                    )


                    fields = error.get(
                        "fields",
                        {}
                    )


                    for field, message in fields.items():

                        st.warning(
                            f"{field}: {message}"
                        )


                except ValueError:

                    st.error(
                        "❌ Backend validation failed."
                    )


            # =================================================
            # OTHER API ERRORS
            # =================================================

            elif response.status_code == 404:

                st.error(
                    "❌ Store or product was not found."
                )


            elif response.status_code == 409:

                st.error(
                    "⚠️ Inventory data is stale or conflicting."
                )


            elif response.status_code == 429:

                st.error(
                    "⚠️ Too many requests. Please try again later."
                )


            elif response.status_code in (500, 503):

                st.error(
                    "⚠️ Forecast service is currently unavailable."
                )


            else:

                st.error(

                    f"❌ Backend returned HTTP "
                    f"{response.status_code}"

                )


        # =====================================================
        # CONNECTION ERROR
        # =====================================================

        except http_requests.exceptions.ConnectionError:

            st.error(

                "❌ Cannot connect to the DFIO backend. "
                f"Please make sure the backend is running at "
                f"{API_BASE_URL}"

            )


        # =====================================================
        # TIMEOUT
        # =====================================================

        except http_requests.exceptions.Timeout:

            st.error(
                "⏱️ Forecast request timed out. Please try again."
            )


        # =====================================================
        # OTHER REQUEST ERROR
        # =====================================================

        except http_requests.exceptions.RequestException as e:

            st.error(
                f"❌ Network error: {e}"
            )