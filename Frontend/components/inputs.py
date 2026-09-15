import streamlit as st

def render_inputs():
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

    return {
        "store_id": store_id,
        "product_id": product_id,
        "weather_condition": weather_condition,
        "inventory_level": inventory_level,
        "discount_rate": discount_rate,
        "units_sold": units_sold,
        "units_ordered": units_ordered,
        "promotion": promotion,
        "epidemic": epidemic,
        "price": price,
        "competitor_pricing": competitor_pricing
    }
