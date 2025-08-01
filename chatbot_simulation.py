import streamlit as st
import matlab.engine
import os

# Set up Streamlit UI
st.set_page_config(page_title="DeepDetect Chatbot", layout="centered")
st.title("🤖 DeepDetect: Deepfake Image Detector")
st.markdown("Upload a facial image and let MATLAB AI predict if it's **Real** or a **Deepfake**.")

uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    st.info("🧠 Starting MATLAB and detecting...")
    try:
        eng = matlab.engine.start_matlab()
        label = eng.predict_deepfake(img_path)
        eng.quit()
        st.success(f"✅ Prediction: **{label.upper()}**")
    except Exception as e:
        st.error(f"❌ Error: {e}")
