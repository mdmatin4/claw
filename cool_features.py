import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("🚀 Cool Streamlit Features Showcase")
st.markdown("Streamlit has a ton of awesome built-in features. Here are a few fun ones to play with!")

# 1. Tabs and Columns Layout
st.header("1. Layouts: Tabs & Columns")
tab1, tab2, tab3 = st.tabs(["Metrics", "Data Editing", "Media & More"])

with tab1:
    st.subheader("Dynamic Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")

with tab2:
    st.subheader("Interactive Data Editor")
    st.markdown("Try editing the table below!")
    df = pd.DataFrame({
        "Task": ["Buy milk", "Walk dog", "Code in Streamlit"],
        "Status": [True, False, True],
        "Priority": ["Low", "High", "Critical"]
    })
    edited_df = st.data_editor(df, num_rows="dynamic")

with tab3:
    st.subheader("Camera Input")
    st.markdown("You can even take a picture directly from your browser!")
    picture = st.camera_input("Take a picture")
    if picture:
        st.image(picture)

st.divider()

# 2. Expanders and Popovers
st.header("2. Hidden Elements")
with st.expander("Click me to expand!"):
    st.write("Surprise! You can hide long text or settings inside expanders to keep your app clean.")
    st.image("https://images.unsplash.com/photo-1542204165-65bf26472b9b?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60")

# 3. Interactive Widgets & Effects
st.header("3. Fun Effects")
if st.button("Trigger a Toast Notification"):
    st.toast('Here is your toast! 🍞', icon='🔥')

if st.button("Make it Snow!"):
    st.snow()

# Native Line Chart (Replaces ECharts)
st.subheader("Interactive Line Chart")
st.markdown("Here is a native Streamlit chart that works perfectly without extra dependencies:")
chart_data = pd.DataFrame(
    [820, 932, 901, 934, 1290, 1330, 1320],
    index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    columns=["Series 1"]
)
st.line_chart(chart_data)

# 4. Progress Bars
st.header("4. Progress Bars")
if st.button("Start Download"):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success("Download Complete!")

