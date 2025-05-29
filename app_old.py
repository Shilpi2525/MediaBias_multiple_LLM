import streamlit as st
import os

from rag import graph_streamer

IMAGE_ADDRESS = "https://bs-uploads.toptal.io/blackfish-uploads/components/open_graph_image/8957316/og_image/optimized/0919_Machines_and_Trust_Lina_Social-ac9acf8ebc252ec57a9a9014f6be62b2.png"

# title
st.title("BiasReader")

# set the image
st.image(IMAGE_ADDRESS)
# add your topic
user_text = st.text_input("Please Add Your Topic")

if user_text:
    st.subheader("Bias Analysis")
    st.write_stream(graph_streamer(user_text))
