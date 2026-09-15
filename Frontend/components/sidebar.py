import streamlit as st

def render_sidebar():
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

        latest_response = st.session_state.get("forecast_response")

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

    return region, category, time_range
