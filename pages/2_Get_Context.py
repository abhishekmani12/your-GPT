import streamlit as st
from wiki import get_details
MODEL = ['spacy','bert']
models=("Spacy (Domain Related Keywords)", "Bert (Generic Keywords)")
model_option=st.sidebar.selectbox("Models", range(len(models)), format_func= lambda x: models[x])
st.sidebar.header("Get more info about important keywords via NER")

deets=[]
tabbers , ner_text = None, None
st.markdown("### Enter text for extraction")
ner_text = st.text_area(" ")

if st.button("Extract") and ner_text :
    with st.spinner("Extracting, Parsing and Fetching Content ..."):
        deets = get_details(ner_text, MODEL[model_option])

    if deets is not None:
        tabbers = st.tabs(deets[1])
        for idx in range(len(deets[1])):
            tabbers[idx].header(deets[1][idx])
            tabbers[idx].write(deets[0].get(deets[1][idx].lstrip(" ")))
    else:
        st.error("No keyword found")

    st.cache_resource.clear()
    st.cache_data.clear()