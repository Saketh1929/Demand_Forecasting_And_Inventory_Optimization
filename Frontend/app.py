import os
from pathlib import Path
import requests
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


API_BASE_URL = os.getenv(
    "DFIO_API_BASE_URL",
    "http://localhost:8000"
).rstrip("/")

FORECAST_ENDPOINT = f"{API_BASE_URL}/api/v1/forecasts"
AGENT_FORECAST_ENDPOINT = f"{API_BASE_URL}/api/forecast"
APPROVAL_ENDPOINT = f"{API_BASE_URL}/api/approve"
STATUS_ENDPOINT = f"{API_BASE_URL}/api/v1/model/status"

API_HEADERS = {}
if os.getenv("DFIO_API_KEY"):
    API_HEADERS["Authorization"] = f"Bearer {os.environ['DFIO_API_KEY']}"
if os.getenv("DFIO_USER_ROLE"):
    API_HEADERS["X-User-Role"] = os.environ["DFIO_USER_ROLE"]

if "forecast_response" not in st.session_state:
    st.session_state.forecast_response = None
if "approval_response" not in st.session_state:
    st.session_state.approval_response = None


def fetch_model_status():
    try:
        response = requests.get(STATUS_ENDPOINT, headers=API_HEADERS, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


model_status = fetch_model_status()
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
    accuracy = latest_response.get("model", {}).get("accuracy") if isinstance(latest_response, dict) and "model" in latest_response else None
    confidence = latest_response.get("confidence_score")

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
            ["Store 1", "Store 2", "Store 3", "Store 4", "Store 5"],
            index=None,
            placeholder="Select store"
        )

    with col2:
        product_id = st.selectbox(
            "Product ID *",
            [f"P{i:04d}" for i in range(1, 21)],
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
            step=5
        )

    with col8:
        units_ordered = st.number_input(
            "Units Ordered *",
            min_value=0,
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
# PRICING
# =========================
with st.container(border=True):

    st.markdown(
        """
        <div class="input-card-title">
            💰 Pricing Inputs
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:#6f8fb5; font-size:14px; margin-top:6px; margin-bottom:10px;">'
        'Own price and competitor price are tracked independently.'
        '</div>',
        unsafe_allow_html=True
    )

    price_col, competitor_col = st.columns(2)

    with price_col:
        price = st.number_input(
            "Price *",
            min_value=0.0,
            max_value=10000.0,
            step=1.0,
            format="%.2f",
        )

    with competitor_col:
        competitor_pricing = st.number_input(
            "Competitor Pricing *",
            min_value=0.0,
            max_value=10000.0,
            step=1.0,
            format="%.2f",
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
if "forecast_response" not in st.session_state:
    st.session_state.forecast_response = None

if test:
    missing_fields = []
    required_values = {
        "Region": region,
        "Category": category,
        "Forecast Period": time_range,
        "Store Name": store_id,
        "Product ID": product_id,
        "Weather Condition": weather_condition,
        "Epidemic": epidemic,
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
            "store_id": f"S{int(store_id.split()[-1]):03d}",
            "product_id": f"P{int(product_id[1:]):04d}",
            "category": category,
            "region": region,
            "inventory_level": int(inventory_level),
            "units_sold": int(units_sold),
            "units_ordered": int(units_ordered),
            "price": float(price),
            "discount": float(discount_rate),
            "competitor_pricing": float(competitor_pricing),
            "weather_condition": weather_condition,
            "promotion": int(promotion),
            "seasonality": "Spring",
            "epidemic": int(epidemic),
            "forecast_period": time_range,
        }

        try:
            with st.spinner("🧠 AI model is generating the forecast..."):
                response = requests.post(
                    AGENT_FORECAST_ENDPOINT,
                    headers=API_HEADERS,
                    json=payload,
                    timeout=30,
                )

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
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to the DFIO backend. Please make sure the backend is running at {API_BASE_URL}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Forecast request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {e}")

forecast_data = st.session_state.forecast_response
if forecast_data is not None:
    data = forecast_data
    predicted_demand = data.get("predicted_demand", "N/A")
    recommended_order = data.get("reorder_quantity", "N/A")
    inventory_gap = data.get("gap", "N/A")
    stock_status = data.get("status", "N/A")
    urgency = data.get("urgency", "N/A")
    stockout_risk = data.get("expected_stockout_risk")
    confidence = data.get("confidence_score")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Predicted Demand", f"{predicted_demand} units")
    with col2:
        st.metric("🛒 Recommended Order", f"{recommended_order} units")
    with col3:
        st.metric("⚠️ Stockout Risk", f"{stockout_risk * 100:.1f}%" if isinstance(stockout_risk, (int, float)) else "N/A")
    with col4:
        st.metric("🎯 Confidence", f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else "N/A")

    st.subheader("📦 Inventory Context")
    context_col1, context_col2, context_col3, context_col4 = st.columns(4)
    context_col1.metric("Inventory Gap", f"{inventory_gap} units")
    context_col2.metric("Stock Status", stock_status)
    context_col3.metric("Urgency", urgency)
    context_col4.metric("Recommended Order", f"{recommended_order} units")

    st.subheader("🧠 Executive Recommendation / AI Analysis")
    st.info(data.get("recommendation", "No recommendation was returned by the backend."))
    st.caption(f"Reasoning source: {data.get('reasoning_source', 'N/A')}")

    accuracy = model_status.get("accuracy") if model_status else None
    accuracy_display = f"{accuracy * 100:.1f}%" if isinstance(accuracy, (int, float)) else "N/A"
    st.info(
        f"🤖 Model: {model_status.get('model_name', 'N/A') if model_status else 'N/A'} | "
        f"Version: {model_status.get('model_version', 'N/A') if model_status else 'N/A'} | "
        f"Accuracy: {accuracy_display} | Request ID: {data.get('forecast_id', 'N/A')} | "
        f"Generated: {data.get('timestamp', 'N/A')}"
    )


# =========================================================
# MANAGER APPROVAL
# =========================================================

latest_response = st.session_state.forecast_response
if latest_response and latest_response.get("requires_approval"):
    st.divider()
    st.subheader("✅ Manager Approval Required")
    st.caption("Review the backend recommendation and submit your decision to the Approval Engine.")

    approval_action = st.selectbox(
        "Decision",
        ["approved", "modified", "rejected"],
        format_func=lambda value: value.title(),
    )
    modified_quantity = None
    if approval_action == "modified":
        modified_quantity = st.number_input(
            "Modified Order Quantity",
            min_value=0,
            value=int(latest_response.get("reorder_quantity", 0)),
            step=1,
        )
    manager_notes = st.text_area("Manager Notes", placeholder="Add review notes")

    if st.button("Submit Manager Decision", type="primary"):
        approval_payload = {
            "forecast_id": latest_response.get("forecast_id", ""),
            "store_id": latest_response.get("store_id", ""),
            "product_id": latest_response.get("product_id", ""),
            "predicted_demand": float(latest_response.get("predicted_demand", 0)),
            "inventory_level": int(latest_response.get("inventory_level", 0)),
            "suggested_reorder_quantity": int(latest_response.get("reorder_quantity", 0)),
            "action": approval_action,
            "modified_quantity": int(modified_quantity) if modified_quantity is not None else None,
            "manager_notes": manager_notes,
        }
        try:
            with st.spinner("Submitting manager decision..."):
                approval_response = requests.post(
                    APPROVAL_ENDPOINT,
                    headers=API_HEADERS,
                    json=approval_payload,
                    timeout=15,
                )
            if approval_response.status_code == 200:
                st.session_state.approval_response = approval_response.json()
                st.success(
                    f"Decision recorded: {approval_action.title()} "
                    f"({st.session_state.approval_response.get('approval_id', 'N/A')})"
                )
            else:
                st.error(f"Approval failed ({approval_response.status_code}): {approval_response.text}")
        except requests.exceptions.RequestException as error:
            st.error(f"Could not submit approval decision: {error}")

if st.session_state.approval_response:
    record = st.session_state.approval_response.get("record", {})
    st.info(
        f"Final decision: {record.get('action', 'N/A').title()} | "
        f"Approval ID: {st.session_state.approval_response.get('approval_id', 'N/A')}"
    )