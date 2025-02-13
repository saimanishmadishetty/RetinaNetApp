import streamlit as st
import base64
import json
import os
from vipas import model, user, exceptions
from PIL import Image
import io
import requests

model_client = model.ModelClient()

# Streamlit app UI
st.title("Image Prediction with Vipas.AI")

# Example image links
example_images = {
    "Example 1": "https://utils.vipas.ai/vipas-images/curated_images/test1.jpg",
    "Example 2": "https://utils.vipas.ai/vipas-images/curated_images/test2.jpg",
    "Example 3": "https://utils.vipas.ai/vipas-images/curated_images/test3.jpg",
    "Example 4": "https://utils.vipas.ai/vipas-images/curated_images/test4.jpg",
    "Upload Your Own": "Upload"
}

selected_example = st.selectbox("Choose an example image", list(example_images.keys()))
image = None

if selected_example in example_images and selected_example != "Upload Your Own":
    image_url = example_images[selected_example]
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
elif selected_example == "Upload Your Own":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

if image:
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Prepare input JSON
    input_body = {
        "inputs": [
            {
                "name": "image",
                "datatype": "BYTES",
                "shape": [1],
                "data": [base64_image]
            }
        ]
    }
    
    if st.button("Predict"):
        try:
            response = model_client.predict(model_id="mdl-3rfp3u0durn9v", input_data=input_body)
            
            if response and response.get("outputs", None):
                output_base64 = response.get("outputs")[0].get("data")[0]
                output_image = Image.open(io.BytesIO(base64.b64decode(output_base64)))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Input Image", use_column_width=True)
                with col2:
                    st.image(output_image, caption="Output Image", use_column_width=True)
            else:
                st.error("No output received from model.")
        except exceptions.ClientException as e:
            st.error(f"Client Exception: {e}")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")
