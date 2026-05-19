import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Random Walk Simulator", page_icon="📈")

st.title("Interactive Random Walk Simulator 🚶‍♂️")
st.markdown("Welcome to this fun little interactive Streamlit app! Adjust the sliders on the left to see what happens.")

st.sidebar.header("Settings ⚙️")
num_steps = st.sidebar.slider("Number of steps", min_value=100, max_value=5000, value=1000, step=100)
num_paths = st.sidebar.slider("Number of paths", min_value=1, max_value=10, value=3, step=1)

if st.button("Generate Random Walk"):
    with st.spinner(f"Simulating {num_paths} paths with {num_steps} steps each..."):
        # Generate random walks
        data = np.random.randn(num_steps, num_paths)
        df = pd.DataFrame(data).cumsum()
        
        # Rename columns for the legend
        df.columns = [f"Path {i+1}" for i in range(num_paths)]
        
        # Display the interactive chart
        st.line_chart(df)
        
        st.success("✅ Simulation complete!")
        st.info("📊 Your random walk is ready for analysis")
        
        # Show some metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Paths", num_paths)
        with col2:
            st.metric("Steps", num_steps)
        with col3:
            st.metric("Status", "Ready")
