import streamlit as st
import base64
import json
import requests
from vipas import model, exceptions

model_client = model.ModelClient()

# Example image links
example_images = {
    "Example 1": "https://utils.vipas.ai/vipas-images/diabetic_retinopathy_test_images/0.jpeg",
    "Example 2": "https://utils.vipas.ai/vipas-images/diabetic_retinopathy_test_images/1.jpeg",
    "Example 3": "https://utils.vipas.ai/vipas-images/diabetic_retinopathy_test_images/2.jpeg",
    "Example 4": "https://utils.vipas.ai/vipas-images/diabetic_retinopathy_test_images/3.jpeg",
}

# Streamlit UI
st.title("Diabetic Retinopathy Detection")
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
    col1, col2 = st.columns([0.5, 0.5])  # Two equal-width columns

    with col1:
        st.image(selected_image, caption="Selected Image", use_column_width=True)

    with col2:
        st.markdown("### Prediction Result:")
        prediction_placeholder = st.empty()  # Placeholder for dynamic updates

    # Predict button centered below both columns
    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
    with col2:
        predict_button = st.button("Predict", use_container_width=True)

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
            response = model_client.predict(model_id="mdl-u28qo4e90ri0a", input_data=input_body)
            print(response)

            # Display prediction output in col2
            if response and response.get("outputs"):
                predicted_class_name = response.get("outputs")[2].get("data")[0]  # Class name
                confidence_scores = response.get("outputs")[1].get("data")  # Confidence scores (array)
                class_description = response.get("outputs")[3].get("data")[0]  # Description

                # Display the class name, confidence score (for the predicted class), and description
                predicted_score = confidence_scores[0]  # Confidence of the predicted class (first element)
                answer_string = f"Predicted Class: {predicted_class_name} \n\n Confidence Score: {predicted_score:.2f} \n\n Description: {class_description}"

                prediction_placeholder.write(answer_string)

            else:
                prediction_placeholder.error("No response received from the model.")

        except exceptions.ClientException as e:
            prediction_placeholder.error(f"Client Exception: {e}")
        except Exception as e:
            prediction_placeholder.error(f"Unexpected Error: {e}")
