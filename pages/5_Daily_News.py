import streamlit as st
from news import get_news
from palmquery import ask_palm
import pandas as pd

st.markdown("### Enter your keyword")

summary_flag=False
@st.cache_resource
def get_cache_news(term, count):
    df=get_news(term, count)
    return df

count = st.sidebar.slider(label="Count", min_value = 1, max_value=100, value=10)
st.sidebar.header("Get your daily news and summarize them")

term = st.text_input(" ")
summary_flag = st.checkbox("Summarize (5)")

if st.button("Get News") and term:
    df = get_cache_news(term, count)
    st.dataframe(df['Text'], hide_index=True, use_container_width=True)

    st.divider()
    if summary_flag:

        with st.spinner("Creating Summaries ..."):
            tabbers = st.tabs([str(i) for i in range(1,6)])
            for idx in range(5):
                tabbers[idx].header(f"{term} - Summary ({idx+1})")
                tabbers[idx].write(ask_palm(df.iloc[idx,1], summary=True))

            



