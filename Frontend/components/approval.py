import streamlit as st
import requests

def render_approval(api_headers, approval_endpoint):
    latest_response = st.session_state.get("forecast_response")
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
                        approval_endpoint,
                        headers=api_headers,
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

    if st.session_state.get("approval_response"):
        record = st.session_state.approval_response.get("record", {})
        st.info(
            f"Final decision: {record.get('action', 'N/A').title()} | "
            f"Approval ID: {st.session_state.approval_response.get('approval_id', 'N/A')}"
        )
