import streamlit as st

def format_units(val):
    if val is None or val == "N/A":
        return "N/A"
    try:
        return f"{int(round(float(val)))} units"
    except (ValueError, TypeError):
        return f"{val} units"

def render_results(model_status):
    forecast_data = st.session_state.get("forecast_response")
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
            st.metric("📦 Predicted Demand", format_units(predicted_demand))
        with col2:
            st.metric("🛒 Recommended Order", format_units(recommended_order))
        with col3:
            st.metric("⚠️ Stockout Risk", f"{stockout_risk * 100:.1f}%" if isinstance(stockout_risk, (int, float)) else "N/A")
        with col4:
            st.metric("🎯 Confidence", f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else "N/A")

        st.subheader("📦 Inventory Context")
        context_col1, context_col2, context_col3, context_col4 = st.columns(4)
        context_col1.metric("Inventory Gap", format_units(inventory_gap))
        context_col2.metric("Stock Status", stock_status)
        context_col3.metric("Urgency", urgency)
        context_col4.metric("Recommended Order", format_units(recommended_order))

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
