import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json
import base64
import io
import time

class_names = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

def postprocess(predictions, original_image):
    # Extract prediction data
    boxes = predictions['detection_boxes']
    scores = predictions['detection_scores']
    classes = predictions['detection_classes']
    num_detections = int(predictions['num_detections'])

    # Draw bounding boxes and class names on the original image
    draw = ImageDraw.Draw(original_image)
    width, height = original_image.size

    try:
        # Use a truetype font
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        # If the truetype font is not available, use the default bitmap font
        font = ImageFont.load_default()

    for i in range(num_detections):
        ymin, xmin, ymax, xmax = boxes[i]
        left, right, top, bottom = xmin * width, xmax * width, ymin * height, ymax * height
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        class_id = int(classes[i])
        class_name = class_names.get(class_id, 'Unknown')
        text = f"{class_name} {scores[i]:.2f}"
        
        # Calculate text size
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        
        # Draw text background
        draw.rectangle([left, top, left + text_width, top + text_height], fill="red")
        draw.text((left, top), text, fill="white", font=font)

    # Convert the image with bounding boxes and class names back to base64
    buffered = io.BytesIO()
    original_image.save(buffered, format="JPEG")
    pred_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return pred_image_base64

# Set the title and description
st.title("RetinaNet Object Detection")
st.markdown("""
    Upload an image and let the RetinaNet model detect objects in it.
    This model can identify a variety of objects.
""")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    vps_model_client = model.ModelClient()
    model_id = "mdl-bosb93njhjc97"
    image = Image.open(uploaded_file)

    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    input_data = img_str

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Input Image")
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.markdown("### Output Image")
        detect_button = st.button('🔍 Detect')
        output_placeholder = st.empty()

        if detect_button:
            with output_placeholder:
                st.markdown("<div style='text-align: center;'><img src='https://i.imgur.com/llF5iyg.gif' width='50'/></div>", unsafe_allow_html=True)
            try:
                api_response = vps_model_client.predict(model_id=model_id, input_data=img_str)
                output_base64 = postprocess(api_response, image)
                output_image_data = base64.b64decode(output_base64)
                result_image = Image.open(io.BytesIO(output_image_data))
                
                output_placeholder.image(result_image, caption='Detected Objects', use_column_width=True)
            except UnauthorizedException:
                st.error("Unauthorized exception")
            except NotFoundException as e:
                st.error(f"Not found exception: {str(e)}")
            except RateLimitExceededException:
                st.error("Rate limit exceeded exception")
            except Exception as e:
                st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle, .stMarkdown, .stButton, .stImage {
            text-align: center;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stImage > img {
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        pre {
            background: #e0f7fa;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid #4CAF50;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)
