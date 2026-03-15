import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

model = load_model("road_crack_model.h5")

classes = ["No Crack", "Small Crack", "Severe Crack"]

st.title("AI Road Crack Detection & Severity System")

uploaded_file = st.file_uploader("Upload Road Image")

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.reshape(img,[1,224,224,3])

    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]

    st.subheader("Prediction Result")
    st.success(result)

