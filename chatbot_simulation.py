import streamlit as st
import matlab.engine
import os

# UI setup
st.set_page_config(page_title="DeepDetect Chatbot", layout="centered")
st.title("🤖 DeepDetect: Deepfake Image Detector")
st.markdown("Upload a facial image and let MATLAB AI predict if it's **Real** or a **Deepfake**.")

# Upload widget
uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save image to disk
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(img_path, caption="Uploaded Image", use_column_width=True)
    st.info("🧠 Starting MATLAB and detecting...")

    try:
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()

        # Absolute path for MATLAB
        abs_path = os.path.abspath(img_path)

        # Call MATLAB function and get result
        label = eng.predict_deepfake(abs_path, nargout=1)

        # Stop engine
        eng.quit()

        # Show prediction
        st.success(f"✅ Prediction: **{label}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
