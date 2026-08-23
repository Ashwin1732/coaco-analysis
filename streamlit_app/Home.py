import streamlit as st
from utils.nav import render_navbar
from utils.theme import apply_theme

st.set_page_config(
    page_title="Cocoa Ripeness Analyser",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_navbar("home")
apply_theme()

# Hero Section
hero_left, hero_right = st.columns([1.2, 1], gap="large")

with hero_left:
    st.write("") # vertical spacing
    st.write("")
    st.markdown('<div class="hero-badge"><span class="badge-dot"></span> SVM + Acoustic & Thermal Fusion</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-bottom: 0;'>Predict cocoa pod<br><span style='color: #FF6B6B;'>ripeness with confidence</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: rgba(255,255,255,0.8); max-width: 600px; margin-bottom: 2rem;'>Upload a thermal image and acoustic tap recording — our SVM classifier extracts signal features and predicts ripeness in seconds.</p>", unsafe_allow_html=True)
    st.page_link("pages/2_Analyser.py", label="Analyse Your Pod ➔", icon="🎯")

with hero_right:
    # Build a native, visually appealing "App Preview" mockup to fill the space
    with st.container(border=True):
        st.markdown("<div style='display: flex; gap: 8px; margin-bottom: 15px;'><div style='width: 12px; height: 12px; border-radius: 50%; background: #FF6B6B;'></div><div style='width: 12px; height: 12px; border-radius: 50%; background: #FFD4D4;'></div></div>", unsafe_allow_html=True)
        st.markdown("<span style='color: rgba(255,255,255,0.5); font-family: \"JetBrains Mono\", monospace; font-size: 0.8rem;'>analysis_results.json</span>", unsafe_allow_html=True)
        st.markdown("### 🍫 Cocoa Pod #1024")
        st.metric("Predicted Ripeness", "Ripe", "Confidence: 96%")
        st.progress(0.96)
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Amplitude", "0.84 V")
        c2.metric("RMS", "0.22 V")
        c3.metric("Power", "0.05 W")

st.divider()

# Pipeline Diagram (Native)
st.markdown("<h3 style='text-align: center; color: #EEEEF0; margin-bottom: 2rem;'>🔍 The Multi-Modal Pipeline</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown("### 🌡️")
        st.markdown("**Thermal Image**")
        st.caption("Surface temperature pattern of the pod")

with col2:
    with st.container(border=True):
        st.markdown("### 🎧")
        st.markdown("**Acoustic Signal**")
        st.caption("Vibration response from tapping the pod")

with col3:
    with st.container(border=True):
        st.markdown("### 🧠")
        st.markdown("**<span style='color: #FF6B6B;'>SVM Fusion Model</span>**", unsafe_allow_html=True)
        st.caption("Combines both signal types into one feature vector for classification")

with col4:
    with st.container(border=True):
        st.markdown("### 📊")
        st.markdown("**Ripeness Prediction**")
        st.caption("Predicted label + confidence score")

st.divider()

# Teaser Section
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'>Curious about the science?</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7);'>Read a detailed breakdown of how we extract features and train the SVM classifier.</p>", unsafe_allow_html=True)
    
    col_empty1, col_link, col_empty2 = st.columns([1.5, 1, 1.5])
    with col_link:
        st.page_link("pages/1_How_It_Works.py", label="Read How It Works ➔", icon="📖", use_container_width=True)

st.write("")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; color: rgba(255,255,255,0.5);">
    <p>Copyright &copy; 2026 Cocoa. Design: <a href="https://templatemo.com" style="color: #FF6B6B;">TemplateMo</a></p>
    <p>Privacy | Terms</p>
</div>
""", unsafe_allow_html=True)
