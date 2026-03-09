import streamlit as st
import numpy as np
from PIL import Image
import joblib

# load models
model_rf = joblib.load("model_rf.pkl")
model_svm = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")

classes = {
    0: "No Tumor",
    1: "Glioma Tumor",
    2: "Meningioma Tumor",
    3: "Pituitary Tumor"

}

st.title("Brain Tumor MRI Classification")

col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = st.file_uploader("Upload MRI")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded MRI", use_column_width=True)

with col2:
    model_choice = st.selectbox(
        "Choose Model",
        ("Random Forest", "SVM")
    )

    # الزرار لتفعيل التصنيف
    if st.button("Classify"):
        if uploaded_file is not None:
            # اختيار الموديل
            model = model_rf if model_choice == "Random Forest" else model_svm

            # preprocessing
            img_gray = img.convert("L")
            img_resized = img_gray.resize((64,64))
            img_array = np.array(img_resized).flatten().reshape(1,-1)
            img_scaled = scaler.transform(img_array)

            # prediction
            prediction = model.predict(img_scaled)
            result = classes[prediction[0]]

            st.subheader("Prediction:")
            st.success(result)
        else:
            st.warning("Please upload an MRI image first!")