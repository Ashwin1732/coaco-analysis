import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import uuid
import hashlib

from utils.model import load_models, thermal_probabilities, acoustic_probabilities, fuse_cnn_probabilities
from utils.signal_processing import parse_acoustic_file, extract_features, downsample_waveform
from utils.audio_generator import write_wav

from utils.nav import render_navbar
from utils.theme import apply_theme

st.set_page_config(
    page_title="Analysis Results — Cocoa Ripeness Analyser",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_navbar("analyser")
apply_theme()

st.title("Cocoa Pod Analysis")
st.markdown("Acoustic waveform, thermal reference, extracted features, and SVM ripeness prediction.")

# 1. Load models
models_state = load_models()

# 2. Uploaders
st.subheader("Upload Batch Samples")

col_t, col_a = st.columns(2)
with col_t:
    thermal_files = st.file_uploader("Thermal images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
with col_a:
    acoustic_files = st.file_uploader("Acoustic files", type=["tdms", "csv"], accept_multiple_files=True)

# Pairing logic
if len(thermal_files) > 0 or len(acoustic_files) > 0:
    if len(thermal_files) != len(acoustic_files):
        st.warning(f"File count mismatch: {len(thermal_files)} thermal images vs {len(acoustic_files)} acoustic files. Please upload matching pairs.", icon="⚠️")
        can_run = False
    else:
        # Assuming paired by order
        pairs = [{"Thermal": t.name, "Acoustic": a.name} for t, a in zip(thermal_files, acoustic_files)]
        st.write("### Sample Pairings")
        st.table(pd.DataFrame(pairs, index=[f"Sample {i+1}" for i in range(len(pairs))]))
        can_run = True

    if 'results' not in st.session_state:
        st.session_state.results = []

    if can_run:
        if st.button("Run Batch Analysis", type="primary"):
            st.session_state.results = []
            
            progress_bar = st.progress(0, text="Starting analysis...")
            
            for i, (t_file, a_file) in enumerate(zip(thermal_files, acoustic_files)):
                progress_bar.progress((i) / len(thermal_files), text=f"Analysing sample {i+1} of {len(thermal_files)}...")
                
                # 1. Process Thermal
                t_bytes = t_file.read()
                img = Image.open(io.BytesIO(t_bytes)).convert("RGB")
                t_probs = thermal_probabilities(img, models_state)
                
                # 2. Process Acoustic
                a_bytes = a_file.read()
                signal = parse_acoustic_file(a_bytes, a_file.name)
                a_probs = acoustic_probabilities(signal.voltage, signal.sample_rate_hz, models_state)
                
                # 3. Fuse and predict
                probabilities = fuse_cnn_probabilities(t_probs, a_probs)
                
                # 4. Features
                feats = extract_features(signal.voltage)
                
                # 5. Generate WAV
                wav_io = io.BytesIO()
                write_wav(signal.voltage, signal.sample_rate_hz, wav_io)
                wav_io.seek(0)
                
                # 6. Downsampled waveform for plotting
                ds_wave = downsample_waveform(signal.voltage, target_points=800)
                
                st.session_state.results.append({
                    "sample_id": i + 1,
                    "thermal_img": img,
                    "wav_bytes": wav_io.read(),
                    "waveform": ds_wave,
                    "prediction": probabilities[0],
                    "probabilities": probabilities,
                    "features": {
                        "amplitude": feats.amplitude,
                        "rms": feats.rms,
                        "power": feats.power
                    }
                })
                
            progress_bar.progress(1.0, text="Analysis complete!")


# 3. Render Results
if 'results' in st.session_state and len(st.session_state.results) > 0:
    st.markdown("---")
    st.subheader("Analysis Results")
    
    summary_data = []
    
    for res in st.session_state.results:
        with st.container(border=True):
            st.markdown(f"### Sample {res['sample_id']}")
            
            # Layout: Left for acoustic, Right for thermal
            col_l, col_r = st.columns([1.5, 1])
            
            with col_l:
                st.markdown("**Acoustic Waveform**")
                st.line_chart(res["waveform"], height=200)
                st.audio(res["wav_bytes"], format="audio/wav")
                
            with col_r:
                st.markdown("**Thermal Reference**")
                st.image(res["thermal_img"], use_container_width=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prediction and possibilities
            col_p, col_f = st.columns([1, 1.5])
            
            with col_p:
                st.metric("Predicted Ripeness", res["prediction"]["label"], f"{res['prediction']['confidence'] * 100:.1f}% confidence")
                
                if res["prediction"]["confidence"] < 0.85:
                    st.markdown("**Other Possibilities**")
                    for p in res["probabilities"][1:]:
                        if p["confidence"] > 0:
                            st.write(f"**{p['label']}** ({p['confidence'] * 100:.1f}%)")
                            st.progress(float(p["confidence"]))
                            
            with col_f:
                st.markdown("**Extracted Features**")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Amplitude", f"{res['features']['amplitude']:.4f} V")
                fc2.metric("RMS", f"{res['features']['rms']:.4f} V")
                fc3.metric("Power", f"{res['features']['power']:.4f}")
                
            # Add to summary
            summary_data.append({
                "Sample": f"Sample {res['sample_id']}",
                "Prediction": res["prediction"]["label"],
                "Confidence": f"{res['prediction']['confidence'] * 100:.1f}%",
                "Amplitude": res["features"]["amplitude"],
                "RMS": res["features"]["rms"],
                "Power": res["features"]["power"]
            })
            
    st.markdown("---")
    st.subheader("Analysis Summary")
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

