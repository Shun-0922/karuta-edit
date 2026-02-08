import streamlit as st
import pandas as pd
import numpy as np

# Set the page configuration
st.set_page_config(page_title="My Streamlit App", layout="wide")

# --- Header Section ---
st.title('Welcome to My Streamlit App こんにちは 👋')
st.write('This is a basic template to get you started with Streamlit.')

# --- Interactive Widget Example ---
st.subheader('Interactive Slider Example')
# Create a slider widget and store its value
slider_value = st.slider('Select a range of values', 0, 100, (25, 75))
st.write(f'You selected a range from {slider_value[0]} to {slider_value[1]}')

# --- Data Display Example ---
st.subheader('DataFrame Display Example')
# Create some sample data
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
# Display the dataframe
st.dataframe(df)

# --- Chart Example ---
st.subheader('Line Chart Example')
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)

# --- Sidebar Example ---
st.sidebar.header("Sidebar Section")
st.sidebar.write("You can add widgets and content here.")
toggle = st.sidebar.checkbox("Toggle me")
if toggle:
    st.sidebar.success("Toggled On!")
else:
    st.sidebar.info("Toggled Off")