
import streamlit as st

import base64
from pathlib import Path

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('yourgpt.png')+"</p>", unsafe_allow_html=True)

#st.columns(3)[1].image('yourgpt.png', output_format='png')
st.sidebar.header('Local and web hosted RAG implementation with a suite of text utility tools')


