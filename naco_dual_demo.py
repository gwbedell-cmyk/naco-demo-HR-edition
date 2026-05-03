import streamlit as st
from bridge import MGEPlusBridge
import numpy as np

st.set_page_config(page_title="INTENTICS • NACO 2026", layout="wide")

# Custom styling
st.markdown("""
<style>
    div.stButton > button {
        background-color: #d4f5d0;
        color: black;
        border-radius: 8px;
        border: 1px solid #000000;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #c6efc1;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([2, 5])
with col1:
    st.image("intentics_logo3.jpg", width=400)
with col2:
    st.markdown("## powered by TEF & MGE+")
    st.markdown("### Real Moral Geometry • Candidate Role Alignment")

st.markdown("**See how aligned two testimonies are — and what it would take to close the gap.**")
st.markdown("---")

# Input columns
col_f, col_i = st.columns(2)

with col_f:
    st.markdown("### Candidate Testimony")
    founder_text = st.text_area(
        "Founder Testimony",
        placeholder="Describe the person you are, what drives you, and what kind of work environment brings out the best in you...",
        height=220,
        key="founder_input"
    )

with col_i:
    st.markdown("### Role & Team Testimony")
    investor_text = st.text_area(
        "Investor Testimony",
        placeholder="Describe the role, the team, and the qualities that would make someone truly excel in this position...",
        height=220,
        key="investor_input"
    )

# Initialize bridge
bridge = MGEPlusBridge()

# Result container
result_container = st.empty()

# Single Show Alignment button
if st.button("Show Alignment", type="primary", key="main_alignment"):
    if not founder_text.strip() or not investor_text.strip():
        with result_container.container():
            st.warning("Please provide both testimonies.")
    else:
        result = bridge.run_dual_analysis(founder_text, investor_text)
        
        with result_container.container():
            st.success(f"**Shared Coherence Score: {result['score']}%**")
            
            st.markdown("### Alignment Gap")
            st.info(result["gap"])
            
            st.markdown("### Recommended Candidate Adjustment")
            st.write(result["founder_adjustment"])
            
            st.markdown("### Recommended Role Adjustment")
            st.write(result["investor_adjustment"])
            
            st.metric("**Projected Improved Coherence**", f"{result['improved_score']}%")

# Footer
st.caption("Intentics • Real Alignment Intelligence • HR Edition • NACO Ottawa 2026")
