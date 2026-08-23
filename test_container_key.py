import streamlit as st
try:
    st.container(key="test")
    print("SUCCESS")
except TypeError as e:
    print("FAILED:", str(e))
