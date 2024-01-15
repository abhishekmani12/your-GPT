import streamlit as st
from load import analyze

st.sidebar.header("Visualize and Analyze tabular data with an interactive module")

REPORT_PATH = "/home/abshk/Desktop/llm/reports/report.html"
st.markdown("### Upload your CSV Dataset")
dataset_path = None
wh = st.empty()
dataset_path = st.file_uploader(" ", accept_multiple_files=False, type=['csv'])

if dataset_path:
    with st.spinner("Analysing"):
        report=analyze(dataset_path)
        report.show_html(REPORT_PATH, open_browser=True, layout='vertical', scale=1.0)
    
    with wh.container():
        st.success("Analysis Completed, Report is saved and will pop up in a new tab.") 
