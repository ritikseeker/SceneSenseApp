import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import torch
import pytesseract
from gtts import gTTS
import tempfile

from environs import Env
env = Env()
env.read_env()

# API Key configuration
api_key = env('API_KEY', None)
if api_key is None:
    raise ValueError("Please enter API Key")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Set Tesseract command path for OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Streamlit Page Configuration
st.set_page_config(page_title="SceneSense 👀", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.main-title { font-size: 50px; font-weight: bold; text-align: center; color: #ADD8E6; margin-top: -18px; }
.subtitle { font-size: 17px; color: #E6E6FA; text-align: center; margin-bottom: 20px; }
.feature-header { font-size: 24px; color: #333; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"> SceneSense </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Let me(AI) help you in perceiving this world! 🗺️ Just like Spider-Man\'s spidey sense, I will scan the environment for objects and will direct how to navigate(?) all through voice. Just upload image of scene in front of you and experience my capabilities. </div>', unsafe_allow_html=True)


# Load Object Detection Model with caching
@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

def detect_objects(image, threshold=0.3, iou_threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    predictions = object_detection_model([img_tensor])[0]
    keep = torch.ops.torchvision.nms(predictions['boxes'], predictions['scores'], iou_threshold)
    
    filtered_predictions = {
        'boxes': predictions['boxes'][keep],
        'labels': predictions['labels'][keep],
        'scores': predictions['scores'][keep],
    }
    return filtered_predictions

def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for label, box, score in zip(predictions['labels'], predictions['boxes'], predictions['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=5)
    return image

def extract_text_from_image(uploaded_file):
    img = Image.open(uploaded_file)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text.strip() or "No text found in the image."

def text_to_speech(text):
    tts = gTTS(text, lang='en', slow=False, tld='com')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")

# Converting image to bytes
def image_to_bytes(file):
    bytes_data = file.getvalue()
    return [{"mime_type": file.type, "data": bytes_data}]

# Function to call the Gemini AI for personalized assistance
def get_assistance_response(input_prompt, image_data):
    system_prompt = """You are a specialized AI that provides accessibility assistance to visually impaired individuals. Visually impaired user will ask you queries and your goal is to provide clear answer with step by step process(where applicable)."""
    
    full_prompt = f"{system_prompt}\n{input_prompt}"
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content([full_prompt, image_data[0]])
    return response.text

# Streamlit UI Design
st.sidebar.image(r"/Users/ritik/Downloads/ai.jpg", use_container_width=True)
st.sidebar.header("Upload")
uploaded_file = st.sidebar.file_uploader("Upload an Image:", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

st.markdown("""
### Features 
- **Real-Time Scene Analysis**: Describes the content of an image, enabling users to understand the scene effectively through text and voice.
- **Object Detection**: Identifies objects and obstacles for enhancing situational awareness.
- **Personalized Assistance**: Provides context-aware suggestions through text and voice.
- **Text & Speech**: text as well as audio outputs.
""")

tab1, tab2, tab3, tab4 = st.tabs(["Real-Time Scene Analysis", "Object Detection", "Personalized Assistance", "Text & Speech"])

# Scene Analysis Tab
with tab1:
    st.subheader("Real-Time Scene Analysis")
    if uploaded_file:
        with st.spinner("Analyzing Image..."):
            image_data = image_to_bytes(uploaded_file)
            user_prompt = "Generate descriptive textual output that interprets the content of the uploaded image to understand the scene effectively."
            response = get_assistance_response(user_prompt, image_data)
            st.write(response)
            text_to_speech(response)

# Object Detection Tab
with tab2:
    st.subheader("Object Detection")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            predictions = detect_objects(image)
            if predictions:
                image_with_boxes = draw_boxes(image.copy(), predictions)
                st.image(image_with_boxes, caption="Objects Identified", use_container_width=True)
            else:
                st.write("No objects detected in the image.")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

# Assistance Tab
with tab3:
    st.subheader("Personalized Assistance")
    if uploaded_file:
        # st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Analyzing for personalized assistance..."):
            image_data = image_to_bytes(uploaded_file)
            user_prompt = "Provide detailed assistance regarding how to navigate through the road/highway/bridge present in the scene in the uploaded image."
            response = get_assistance_response(user_prompt, image_data)
            st.write(response)
            text_to_speech(response)

# Text-to-Speech Tab
with tab4:
    st.subheader("Text Extraction & Speech")
    if uploaded_file:
        text = extract_text_from_image(uploaded_file)
        st.write(f"Extracted Text: {text}")
        if text:
            text_to_speech(text)
