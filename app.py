import streamlit as st
import base64
import io
import json
import requests
from vipas import model, exceptions
from PIL import Image

# Initialize model client
model_client = model.ModelClient()

# Streamlit app UI
st.title("Weed detection in crops")

# Example image links
example_images = {
    "Example 1": "https://utils.vipas.ai/vipas-images/weed_detection/test1.jpeg",
    "Example 2": "https://utils.vipas.ai/vipas-images/weed_detection/test2.jpeg",
    "Example 3": "https://utils.vipas.ai/vipas-images/weed_detection/test3.jpeg",
    "Example 4": "https://utils.vipas.ai/vipas-images/weed_detection/test4.jpeg",
    "Upload Your Own": "Upload"
}

# Select image source
selected_example = st.selectbox("Choose an example image", list(example_images.keys()))

# Image placeholders
image = None
output_image = None

# Fetch example image or uploaded image
if selected_example in example_images and selected_example != "Upload Your Own":
    image_url = example_images[selected_example]
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
elif selected_example == "Upload Your Own":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# Display input image immediately after selection/upload
if image:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_column_width=True)

    # Placeholder for the output image (initially empty)
    output_container = col2.empty()

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

    # Predict button (placed below the images to maintain layout)
    if st.button("Predict"):
        try:
            response = model_client.predict(model_id="mdl-ywivo8k09baqt", input_data=input_body)

            if response and response.get("outputs", None):
                output_data = response["outputs"][0].get("data", [])[0]  # Extract JSON string
                parsed_data = json.loads(output_data)  # Parse JSON string
                output_base64 = parsed_data[0]["annotated_image_base64"]  # Extract base64 string

                output_image = Image.open(io.BytesIO(base64.b64decode(output_base64)))

                # Update output container with the predicted output image
                output_container.image(output_image, caption="Output Image", use_column_width=True)
            else:
                st.error("No output received from model.")
        except exceptions.ClientException as e:
            st.error(f"Client Exception: {e}")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")
