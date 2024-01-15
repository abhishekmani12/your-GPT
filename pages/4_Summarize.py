import streamlit as st
from geminiquery import ask_gemini


st.sidebar.header("Get a 10 points summary for any text content")
st.markdown("### Enter text for summarization")
summary , unsum_text = None, None

unsum_text = st.text_area(" ")

if st.button("Summarize") and unsum_text:
    with st.spinner("Loading ..."):
        summary = ask_gemini(unsum_text, summary = True)

    st.markdown(summary)


