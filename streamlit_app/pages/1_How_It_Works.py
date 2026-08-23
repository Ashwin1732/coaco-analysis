import streamlit as st
from utils.nav import render_navbar
from utils.theme import apply_theme

st.set_page_config(
    page_title="How It Works - Cocoa Ripeness Analyser",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_navbar("how-it-works")
apply_theme()

# Breadcrumb
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.5); font-family: \"JetBrains Mono\", monospace; font-size: 0.85rem;'>Home <span style='opacity: 0.4; margin: 0 5px;'>/</span> <span style='color: #EEEEF0;'>How It Works</span></p>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #FF6B6B; margin-bottom: 0;'>Workflow</h4>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-top: 0;'>From upload to prediction<br>in four steps</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7); max-width: 600px; margin: 0 auto 3rem auto;'>Combine thermal imaging and acoustic tapping for accurate cocoa pod ripeness classification.</p>", unsafe_allow_html=True)

# Create a 2x2 grid for the workflow steps
row1_col1, row1_col2 = st.columns(2, gap="large")

with row1_col1:
    with st.container(border=True):
        st.markdown("<span style='color: #FF6B6B; font-family: \"JetBrains Mono\", monospace; font-size: 1.2rem; font-weight: bold;'>01</span>", unsafe_allow_html=True)
        st.markdown("### 📤 Upload Inputs")
        st.caption("Submit a thermal image of the cocoa pod alongside its acoustic tap recording (CSV waveform).")
        st.write("")
        # Visual Mockup for Upload
        st.markdown("""
        <div style="border: 1px dashed rgba(255,255,255,0.2); border-radius: 8px; padding: 15px; text-align: center; background: rgba(0,0,0,0.2);">
            <div style="font-size: 1.5rem; margin-bottom: 5px;">📁</div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.7); font-family: 'JetBrains Mono', monospace;">pod_1024_thermal.jpg</div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.7); font-family: 'JetBrains Mono', monospace;">pod_1024_acoustic.csv</div>
        </div>
        """, unsafe_allow_html=True)

with row1_col2:
    with st.container(border=True):
        st.markdown("<span style='color: #FF6B6B; font-family: \"JetBrains Mono\", monospace; font-size: 1.2rem; font-weight: bold;'>02</span>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Extract Features")
        st.caption("The system computes amplitude, RMS, and power features from the acoustic signal.")
        st.write("")
        # Visual Mockup for Features
        c1, c2, c3 = st.columns(3)
        c1.metric("Amplitude", "0.84 V", border=True)
        c2.metric("RMS", "0.22 V", border=True)
        c3.metric("Power", "0.05 W", border=True)

st.write("")

row2_col1, row2_col2 = st.columns(2, gap="large")

with row2_col1:
    with st.container(border=True):
        st.markdown("<span style='color: #FF6B6B; font-family: \"JetBrains Mono\", monospace; font-size: 1.2rem; font-weight: bold;'>03</span>", unsafe_allow_html=True)
        st.markdown("### 🧠 SVM Prediction")
        st.caption("A trained SVM classifier fuses thermal and acoustic embeddings to predict ripeness class with a confidence score.")
        st.write("")
        # Visual Mockup for Fusion
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255,107,107,0.05); border: 1px solid rgba(255,107,107,0.2); padding: 15px; border-radius: 8px;">
            <div style="text-align: center; color: rgba(255,255,255,0.8); font-size: 0.85rem;">[ Thermal ]<br>[ Acoustic ]</div>
            <div style="color: #FF6B6B;">➔</div>
            <div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-weight: bold;">SVM<br>Kernel</div>
            <div style="color: #FF6B6B;">➔</div>
            <div style="text-align: center; color: #4CAF50; font-weight: bold;">RIPE</div>
        </div>
        """, unsafe_allow_html=True)

with row2_col2:
    with st.container(border=True):
        st.markdown("<span style='color: #FF6B6B; font-family: \"JetBrains Mono\", monospace; font-size: 1.2rem; font-weight: bold;'>04</span>", unsafe_allow_html=True)
        st.markdown("### 📊 View Results")
        st.caption("Inspect the predicted label, confidence percentage, extracted features, and interactive waveform visualization.")
        st.write("")
        # Visual Mockup for Results
        st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-bottom: 5px;'>Confidence: 96%</div>", unsafe_allow_html=True)
        st.progress(0.96)
        st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.85rem; color: #4CAF50; margin-top: 5px;'>✓ Classification Successful</div>", unsafe_allow_html=True)

st.divider()

col_empty1, col_link, col_empty2 = st.columns([1, 1, 1])
with col_link:
    st.page_link("pages/2_Analyser.py", label="Try the Analyser ➔", icon="🎯")
    
# Footer
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; color: rgba(255,255,255,0.5);">
    <p>Copyright &copy; 2026 Cocoa. Design: <a href="https://templatemo.com" style="color: #FF6B6B;">TemplateMo</a></p>
    <p>Privacy | Terms</p>
</div>
""", unsafe_allow_html=True)
