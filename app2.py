import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd

# ===========================
# 1) Load pre-trained models
# ===========================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

MODEL_PATH_B0 = r"C:\Users\tenzi\Desktop\fastapi_app\EfficientNetB0_model.keras"
MODEL_PATH_V2 = r"C:\Users\tenzi\Desktop\fastapi_app\EfficientNetV2S_model.keras"

model_b0 = load_model(MODEL_PATH_B0)
model_v2 = load_model(MODEL_PATH_V2)

# ===========================
# 2) Class names
# ===========================
class_names = ['Irish_Terrier', 'Tibetan_Terrier', 'Boxer']

# ===========================
# 3) Streamlit App
# ===========================
st.title("🐾 Dog Breed Classifier with EfficientNetB0 & EfficientNetV2")

uploaded_file = st.file_uploader("Upload an image of a dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))  # EfficientNetB0 & V2 default input size
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # ===========================
    # 4) Predict with EfficientNetB0
    # ===========================
    preds_b0 = model_b0.predict(img_array)
    predicted_index_b0 = np.argmax(preds_b0)
    predicted_label_b0 = class_names[predicted_index_b0]
    confidence_b0 = preds_b0[0][predicted_index_b0]


        # Preprocess image
    img_resized = img.resize((384, 384))  # EfficientNetB0 & V2 default input size
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # ===========================
    # 5) Predict with EfficientNetV2
    # ===========================
    preds_v2 = model_v2.predict(img_array)
    predicted_index_v2 = np.argmax(preds_v2)
    predicted_label_v2 = class_names[predicted_index_v2]
    confidence_v2 = preds_v2[0][predicted_index_v2]

    # ===========================
    # 6) Display results side by side
    # ===========================
    st.subheader("Predictions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**EfficientNetB0**")
        st.success(f"Predicted: **{predicted_label_b0}**")
        st.info(f"Confidence: {confidence_b0:.2%}")
        proba_dict_b0 = {class_names[i]: float(preds_b0[0][i]) for i in range(len(class_names))}
        st.table(pd.DataFrame.from_dict(proba_dict_b0, orient='index', columns=["Probability"]))

    with col2:
        st.markdown("**EfficientNetV2**")
        st.success(f"Predicted: **{predicted_label_v2}**")
        st.info(f"Confidence: {confidence_v2:.2%}")
        proba_dict_v2 = {class_names[i]: float(preds_v2[0][i]) for i in range(len(class_names))}
        st.table(pd.DataFrame.from_dict(proba_dict_v2, orient='index', columns=["Probability"]))
