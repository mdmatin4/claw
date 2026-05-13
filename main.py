import streamlit as st

st.set_page_config(page_title="My App Portfolio", page_icon="✨")

st.sidebar.title("Navigation")

# Define the pages linking to your existing files
random_walk_page = st.Page("streamlit_app.py", title="Random Walk Simulator", icon="📈")
chatbot_page = st.Page("chatbot_app.py", title="SnarkyBot", icon="🤖")
features_page = st.Page("cool_features.py", title="Cool Features Showcase", icon="🚀")
pdf_page = st.Page("pdf_viewer.py", title="PDF Viewer", icon="📄")

# Set up the navigation menu
pg = st.navigation([random_walk_page, chatbot_page, features_page, pdf_page])

# Run the selected page
pg.run()
