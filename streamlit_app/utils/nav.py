import streamlit as st

def render_navbar(active_page: str):
    # Hide Streamlit's default sidebar entirely
    st.markdown(
        """
        <style>
            [data-testid="collapsedControl"] {
                display: none;
            }
            [data-testid="stSidebar"] {
                display: none;
            }
            
            /* Target the specific keyed container */
            .st-key-cocoa_navbar {
                background: rgba(30, 30, 36, 0.7); /* Fallback */
                border-bottom: 1px solid rgba(255, 107, 107, 0.2);
                padding: 12px 24px;
                border-radius: 0 0 16px 16px;
                margin-bottom: 2rem;
            }
            
            @supports (backdrop-filter: blur(16px)) {
                .st-key-cocoa_navbar {
                    background: rgba(30, 30, 36, 0.4);
                    backdrop-filter: blur(16px);
                    -webkit-backdrop-filter: blur(16px);
                }
            }
            
            .st-key-cocoa_navbar div[data-testid="stHorizontalBlock"] {
                align-items: center;
                justify-content: space-between;
            }
            
            .nav-brand {
                font-size: 1.5rem;
                font-weight: 800;
                color: #fff;
                text-decoration: none;
                letter-spacing: -0.02em;
                transition: opacity 0.2s ease;
            }
            
            .nav-brand:hover {
                opacity: 0.85;
            }
            
            /* Cocoa Brown specific accent for the dot */
            .nav-brand span {
                color: #8B5A2B;
            }
            
            /* Streamlit page links inject inside divs, we want to style them */
            .st-key-cocoa_navbar [data-testid="stPageLink-NavLink"] {
                color: rgba(255, 255, 255, 0.7) !important;
                font-weight: 600 !important;
                text-decoration: none !important;
                background: none !important;
                border: none !important;
                box-shadow: none !important;
                transition: color 0.15s ease !important;
            }
            
            .st-key-cocoa_navbar [data-testid="stPageLink-NavLink"]:hover p {
                color: #FF6B6B !important;
            }
            
            /* Active link tracking via injected sibling marker */
            span#active-link-marker + div [data-testid="stPageLink-NavLink"] p {
                color: #FF6B6B !important;
                position: relative;
            }
            
            span#active-link-marker + div [data-testid="stPageLink-NavLink"] p::after {
                content: '';
                position: absolute;
                bottom: -4px;
                left: 0;
                width: 100%;
                height: 2px;
                background: #FF6B6B;
                border-radius: 2px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.container(key="cocoa_navbar"):
        col_brand, col_links = st.columns([1, 2])
        
        with col_brand:
            logo_svg = """
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 36" fill="none" role="img" aria-label="Cocoa" width="120" style="margin-top: 4px;">
              <defs>
                <linearGradient id="podGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stop-color="#FF8585"/>
                  <stop offset="100%" stop-color="#FF6B6B"/>
                </linearGradient>
              </defs>
              <ellipse cx="18" cy="18" rx="11" ry="14" fill="url(#podGrad)"/>
              <path d="M18 5 C14 5 12 10 12 18 C12 26 14 31 18 31 C22 31 24 26 24 18 C24 10 22 5 18 5 Z" stroke="#FFD4D4" stroke-width="1.2" fill="none" opacity="0.5"/>
              <path d="M18 8 C18 8 16 14 18 20 C20 14 18 8 18 8 Z" fill="#1E1E24" opacity="0.25"/>
              <text x="38" y="24" font-family="'Plus Jakarta Sans', system-ui, sans-serif" font-size="20" font-weight="800" fill="#EEEEF0" letter-spacing="-0.04em">Cocoa</text>
            </svg>
            """
            st.markdown(f'<div class="nav-brand">{logo_svg}</div>', unsafe_allow_html=True)
            
        with col_links:
            # We place page links next to each other
            c1, c2, c3 = st.columns(3)
            with c1:
                if active_page == "home": st.markdown('<span id="active-link-marker"></span>', unsafe_allow_html=True)
                st.page_link("Home.py", label="Home")
            with c2:
                if active_page == "how-it-works": st.markdown('<span id="active-link-marker"></span>', unsafe_allow_html=True)
                st.page_link("pages/1_How_It_Works.py", label="How It Works")
            with c3:
                if active_page == "analyser": st.markdown('<span id="active-link-marker"></span>', unsafe_allow_html=True)
                st.page_link("pages/2_Analyser.py", label="Analyser")
