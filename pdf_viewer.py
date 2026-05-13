import streamlit as st
import base64

st.title("📄 PDF Viewer")
st.markdown("Browse and view your local PDF files right here in the app!")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"Successfully loaded: {uploaded_file.name}")
    
    # Read the file to bytes
    bytes_data = uploaded_file.getvalue()
    
    # Encode as base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
    # Embed the PDF using HTML iframe
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    
    st.markdown(pdf_display, unsafe_allow_html=True)
