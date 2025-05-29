import streamlit as st
import os
from rag import graph_streamer

IMAGE_ADDRESS = "https://bs-uploads.toptal.io/blackfish-uploads/components/open_graph_image/8957316/og_image/optimized/0919_Machines_and_Trust_Lina_Social-ac9acf8ebc252ec57a9a9014f6be62b2.png"

st.title("BiasReader")
st.image(IMAGE_ADDRESS)

# Dropdowns to choose LLMs
model_options = ['GPT-4o', 'Gemini-Pro', 'Claude-3', 'DeepSeek', 'Mixtral']

bias_model = st.selectbox("Select Model for Bias Detection", model_options, index=0)
subbias_model = st.selectbox("Select Model for Subcategory Detection", model_options, index=1)

user_text = st.text_input("Please Add Your Topic")

if user_text:
    st.subheader("Bias Analysis")
    st.write_stream(graph_streamer(user_text, bias_model, subbias_model))
