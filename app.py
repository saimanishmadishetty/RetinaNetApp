import streamlit as st
import base64
import json
import os
import requests
from vipas import model, exceptions

model_client = model.ModelClient()

# Example image links
example_images = {
    "Example 1": "https://utils.vipas.ai/vipas-images/curated_images_for_skin_cancer/example1.jpg",
    "Example 2": "https://utils.vipas.ai/vipas-images/curated_images_for_skin_cancer/example2.jpg",
    "Example 3": "https://utils.vipas.ai/vipas-images/curated_images_for_skin_cancer/example3.jpg",
    "Example 4": "https://utils.vipas.ai/vipas-images/curated_images_for_skin_cancer/example4.jpg"
}

# Streamlit UI
st.title("Skin Cancer Classification")
st.write("Choose an option to get a prediction.")

# Dropdown for selecting input type
option = st.selectbox("Select Image Source:", ["Upload an Image", "Use Example Image"])
selected_image = None

if option == "Use Example Image":
    image_choice = st.selectbox("Choose an Example Image:", list(example_images.keys()))
    selected_image = example_images[image_choice]
elif option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        selected_image = uploaded_file

if selected_image:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if isinstance(selected_image, str):
            st.image(selected_image, caption="Selected Image", use_column_width=True)
        else:
            st.image(selected_image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        predict_button = st.button("Predict")
        if predict_button:
            try:
                # Read and encode the image in base64
                if isinstance(selected_image, str):
                    response = requests.get(selected_image)
                    base_64_string = base64.b64encode(response.content).decode("utf-8")
                else:
                    base_64_string = base64.b64encode(selected_image.read()).decode("utf-8")
                
                # Create input JSON body
                input_body = {
                    "inputs": [
                        {
                            "name": "image_base64",
                            "datatype": "BYTES",
                            "shape": [1],
                            "data": [base_64_string]
                        }
                    ]
                }
                
                # Send prediction request
                response = model_client.predict(model_id="mdl-txpby5w7p3jzc", input_data=input_body)
                
                # Display prediction output
                if response and response.get("outputs"):
                    st.success(f'Prediction Result:{response.get("outputs")[0].get("data")[0]}')
                else:
                    st.error("No response received from the model.")
                
            except exceptions.ClientException as e:
                st.error(f"Client Exception: {e}")
            except Exception as e:
                st.error(f"Unexpected Error: {e}")
