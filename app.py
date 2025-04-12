

import streamlit as st
from real_time_detection import run_detection

st.title("Sign Language Detection")

st.markdown("### Click the button below to start real-time detection using your webcam.")

if st.button("Start Detection"):
    st.info("Press 'Q' on the keyboard to stop the webcam.")
    run_detection()
