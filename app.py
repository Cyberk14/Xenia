import streamlit as st
import pandas as pd
import numpy as np

# Title of your GUI
st.title("🚀 Lightning AI Project GUI")

# Text inputs and sidebars
st.sidebar.header("Settings")
user_name = st.sidebar.text_input("Enter your name:", "Developer")

# Interactive layout element
st.write(f"Welcome back, **{user_name}**!")

# A button interactive element
if st.button("Click to run a quick test"):
    st.success("Test ran successfully!")
    
# Example display of data visualization
st.subheader("Model Metric Overview")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['Loss', 'Accuracy', 'Validation']
)
st.line_chart(chart_data)
