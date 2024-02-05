import streamlit as st
import streamlit.components.v1 as components
from load import analyze
import webbrowser

st.sidebar.header("Visualize and Analyze tabular data with an interactive module")

st.markdown("### Upload your CSV Dataset")
dataset_path = None
wh = st.empty()
dataset_path = st.file_uploader(" ", accept_multiple_files=False, type=['csv'])


if dataset_path:
    fname = dataset_path.name.split(".")
    REPORT_PATH = f"reports/{fname[0]}.html"
    with st.spinner("Analysing"):
        report=analyze(dataset_path)
        report.show_html(REPORT_PATH, layout='vertical', scale=1.0)
        
    with wh.container():
        st.success("Analysis Completed, Report is saved") 

    HtmlFile = open(REPORT_PATH, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    
    components.html(source_code, width = 1070, height = 600, scrolling=True)
