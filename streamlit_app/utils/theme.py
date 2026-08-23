import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Approximating Glassmorphism */
        div[data-testid="stContainer"] > div,
        div[data-testid="stMetric"] {
            background: rgba(42, 42, 50, 0.4);
            border: 1px solid rgba(255, 107, 107, 0.2);
            border-radius: 16px !important;
            padding: 1.25rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            color: #FF6B6B;
            font-weight: 700;
        }
        
        /* Links and buttons overrides outside navbar */
        [data-testid="stPageLink-NavLink"] p {
            font-weight: 600;
        }
        
        hr {
            border-color: rgba(255, 107, 107, 0.2) !important;
        }
        
        /* Hero Badge */
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            background: rgba(255, 107, 107, 0.1);
            color: #FF6B6B;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 16px;
            border: 1px solid rgba(255, 107, 107, 0.2);
        }
        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #FF6B6B;
        }
        
        /* Fixed background blobs */
        .page-bg-blobs {
            position: fixed;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            overflow: hidden;
        }
        .bg-blob {
            position: absolute;
            border-radius: 50%;
            background: radial-gradient(
                circle,
                rgba(255, 107, 107, 0.15) 0%,
                transparent 70%
            );
            filter: blur(40px);
        }
        .blob-1 {
            top: -10%;
            left: -10%;
            width: 50vw;
            height: 50vw;
            max-width: 600px;
            max-height: 600px;
        }
        .blob-2 {
            bottom: 10%;
            right: -5%;
            width: 60vw;
            height: 60vw;
            max-width: 800px;
            max-height: 800px;
        }
    </style>
    <div class="page-bg-blobs">
        <div class="bg-blob blob-1"></div>
        <div class="bg-blob blob-2"></div>
    </div>
    """, unsafe_allow_html=True)
