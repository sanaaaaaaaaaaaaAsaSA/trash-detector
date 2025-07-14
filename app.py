import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("trash_model.h5")
class_names = ['glass','metal','paper','plastic','wet']

st.title("Trash Type Image Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img)
    arr = np.array(img)/255.0
    pred = model.predict(arr[np.newaxis,...])[0]
    st.write(f"Prediction: {class_names[np.argmax(pred)]} ({max(pred)*100:.2f}%)")
