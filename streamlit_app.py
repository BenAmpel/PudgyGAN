import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from util import *

#@st.experimental_memo
@st.experimental_singleton
def load_gen():
    model = load_model('generator')
    model.summary()  # included to make it visible when model is reloaded
    return model

st.set_page_config(
    page_title = "PudgyGAN",
    initial_sidebar_state="collapsed",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    generator_network = load_gen()
    st.sidebar.title("About")
    st.sidebar.info("This a project conducted for the MIS 596A: Deep Learning course at the University of Arizona")
    #img = plt.imread('./images/Eller1.png')
    #st.sidebar.image(img)
    st.title('PudgyGAN')
    st.write("The final project for MIS 596A, conducted by Benjamin Ampel and Ryan Ott")
    img1 = plt.imread('./images/Pudgy.jpg')
    st.image(img1, width=224)
    page = st.selectbox("Choose a page", ["Homepage", "Introduction", "Dataset & Pre-Processing", "PudgyGAN Architecture", "Experiments & Results", "Visual Comparisons", "Future Directions", "Generate Pudgy"])
    #-----------------------------------------------------------
    if page == "Homepage":
        st.header("Welcome to the PudgyGAN website")
        st.write("Using the select box above, you can use the model to generate images, and learn more about how this project was conducted")

    #-----------------------------------------------------------
    if page == "Introduction":
        st.header("Introduction")
        NFTimg = plt.imread('./images/NFTimg.jpg')
        st.image(NFTimg, width=512)
        st.subheader("Non-Fungible Tokens (NFTs) are a type of unique digital asset that is secured with blockchain technology.")
        st.subheader("NFTs are becoming a popular investment, with a predicted market cap of $80 billion by 2025 (Canny, 2022).")
        st.subheader("Problem: NFTs are difficult to create, requiring manual drawing of new options.")
        st.subheader("Proposed Solution: A generative adversarial network (GAN) that can generate highly stylized versions of NFTs for consumers.")
    #-----------------------------------------------------------
    if page == "Dataset & Pre-Processing":
        st.header("Dataset & Pre-Processing")
        PudgyEx = plt.imread('./images/PudgyExample.jpg')
        st.image(PudgyEx, width=512)
        st.write(":heavy_minus_sign:" * 34)
        st.header("Pudgy Penguin Dataset")
        st.subheader("Pudgy Penguins are a popular NFT collection with a current market cap of $18,346,394.16 as of March 10, 2022 (CoinGecko, 2022).")
        st.subheader("There are currently 8,888 unique Pudgy Penguins")
        st.subheader("Collection Method: OpenSea API, store each image in a non-relational database (e.g., MongoDB). ")
        st.write(":heavy_minus_sign:" * 34)
        st.header("Data Augmentation")
        DataAug = plt.imread('./images/DataAug.JPG')
        st.image(DataAug, width=800)
        st.subheader("To increase the size of our dataset, extant literature suggests augmenting our images (Khalifa et al., 2021). ")
        st.subheader("There are three types of classical augmentation we can perform on our dataset: (1) geometric, (2) photometric, and (3) random erasing.")
        st.write("Geometric augmentations include includes flipping, rotating, shearing, cropping, and translating the image (Vyas et al., 2018).")
        st.write("Photometric augmentations include color space shifting, adding image filters, and adding random noise.")
        st.write("Random erasing augmentations include deleting random parts of the image (Zhong et al., 2020). All these techniques have shown to improve image-based deep learning tasks.")
        st.subheader("After pre-processing, our dataset has been increased from 8,888 to 60,000")
    #-----------------------------------------------------------
    if page == "PudgyGAN Architecture":
        st.header("PudgyGAN Architecture")
        Ganimg = plt.imread('./images/GAN.jpg')
        st.image(Ganimg, width=800)
        st.subheader("We employ the Generative Adversarial Network (GAN) deep learning architecture introduced by Goodfellow et al. (2004).")
        st.subheader("The GAN feeds images from the real dataset and images generated by a deep learning model into a discriminator model. ")
        st.subheader("The discriminator determines if the image is real or generated.")
        st.write(":heavy_minus_sign:" * 34)

        st.header("Deep Convolutional GAN (DCGAN)")
        DCGAN = plt.imread('./images/DCGAN.png')
        st.image(DCGAN, width=1200)
    #-----------------------------------------------------------
    if page == "Experiments & Results":
        st.header("Experiments & Results")
    #-----------------------------------------------------------
    if page == "Visual Comparisons":
        st.header("Visual Comparisons")
    #-----------------------------------------------------------
    if page == "Future Directions":
        st.header("Future Directions")
    #-----------------------------------------------------------
    if page == "Generate Pudgy":
        st.header("Use the button below to make your own automatically generated Pudgy Penguin!")
        submit = st.button('Click To Generate Pudgy')
        if submit:
            PudgyImage = show_generator_results(generator_network)
            st.pyplot(PudgyImage)





if __name__ == "__main__":
    main()
