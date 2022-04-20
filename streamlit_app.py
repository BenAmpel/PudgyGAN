import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

from IPython.core.display import display, HTML
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from keras.models import Model, load_model
from util import *

st.cache(suppress_st_warning=True)  # 👈 Changed this
generator_network = load_model('generator')

st.set_page_config(
    page_title = "PudgyGAN",
    initial_sidebar_state="collapsed"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    st.sidebar.title("About")
    st.sidebar.info("This a project conducted for the MIS 596A: Deep Learning course at the University of Arizona")
    #img = plt.imread('./images/Eller1.png')
    #st.sidebar.image(img)
    st.title('PudgyGAN')
    img1 = plt.imread('./images/Pudgy.jpg')
    st.image(img1, width=224)
    page = st.selectbox("Choose a page", ["Homepage", "Generate Pudgy"])
    #-----------------------------------------------------------
    if page == "Homepage":
        st.header("Welcome to the PudgyGAN website")
        st.write("Using the select box above, you can use the model to generate images, and learn more about how this project was conducted")
    #-----------------------------------------------------------
    if page == "Generate Pudgy":
        st.header("Use the button below to make your own automatically generated Pudgy Penguin!")
        submit = st.button('Click To Generate Pudgy')
        if submit:
            PudgyImage = show_generator_results(generator_network)
            st.pyplot(PudgyImage)

if __name__ == "__main__":
    main()
