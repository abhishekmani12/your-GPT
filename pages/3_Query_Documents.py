import streamlit as st
import os
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from load import embed
from query import get_answer, get_pipe
load_dotenv()
model_type=os.getenv('model')

st.markdown("### Upload your docs and start querying")

models=("Gemini", "Palm", "GPT4All", "Mistral - 7B", "Llama - 7B")
model_option=st.sidebar.selectbox("Models", range(len(models)), format_func= lambda x: models[x])


st.sidebar.header("Ask queries based on your supplied text documents via a Vector Database")

@st.cache_resource(show_spinner="Loading RAG Pipeline")
def get_cache_rqa(model_type):
    return get_pipe(model_type)

rqa = get_cache_rqa(model_type=model_type)

doc_obj, scan_flag = None, False
wh=st.empty()

with st.form('embed_form'):
    
    doc_obj = st.file_uploader(" ", accept_multiple_files=True, type=['pdf'])
    scan_flag = st.checkbox("Enable OCR (For scanned docs)")

    submitted = st.form_submit_button("Embed")
    if submitted and doc_obj:
        ext=doc_obj[0].name.split(".")

        with st.spinner("Embedding and Storing in Vector Database ..."):
            with NamedTemporaryFile(dir='.', suffix=f".{ext[1]}") as f:
                f.write(doc_obj[0].getbuffer())
                if scan_flag:
                    embed(f.name, scanned=True) 
                else:
                    embed(f.name, scanned=False)
        with wh.container():
                st.success("Document Stored Successfully")
        
        st.cache_resource.clear()
        st.cache_data.clear()

        rqa = get_cache_rqa(model_type=model_type)

#USER
                
response = None
if prompt := st.chat_input():
    st.chat_message("user").markdown(prompt)

    with st.spinner("Loading ..."):
        response=get_answer(query=prompt, RQA=rqa, palm=True)
    response = f"LLM:\n {response[1]}"
    
    #LLM
    with st.chat_message("Assistant"):
        st.markdown(response)