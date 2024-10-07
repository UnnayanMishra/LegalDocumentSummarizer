import gradio as gr
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
from transformers import pipeline, AutoTokenizer, BertLMHeadModel
from googletrans import Translator
from matplotlib import pyplot as plt

# Step 1: Convert to grayscale and binarize the image
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarize_image(gray_image):
    _, im_bw = cv2.threshold(gray_image, 200, 230, cv2.THRESH_BINARY)
    return im_bw

# Step 2: Noise removal
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.medianBlur(image, 3)
    return image

# Step 3: Deskewing the image
def get_skew_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    min_area_rect = cv2.minAreaRect(largest_contour)
    angle = min_area_rect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def deskew(image):
    angle = get_skew_angle(image)
    return rotate_image(image, -angle)

# Step 4: OCR using Tesseract
def perform_ocr(image):
    return pytesseract.image_to_string(image)

# Step 5: Summarization using transformers
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = BertLMHeadModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

def generate_summary(input_text, max_length=500):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 6: Translation using Google Translate
translator = Translator()

def process_image(image):
    # Convert to OpenCV format
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Preprocessing
    gray_image = grayscale(img_cv2)
    binarized_image = binarize_image(gray_image)
    noise_removed_image = noise_removal(binarized_image)
    deskewed_image = deskew(noise_removed_image)

    # Perform OCR
    ocr_result = perform_ocr(deskewed_image)
    
    # Generate Summary
    summary = summarizer(ocr_result, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

    # Translation to Hindi, Marathi, Bengali
    translated_hi = translator.translate(summary, src='en', dest='hi').text
    translated_mr = translator.translate(summary, src='en', dest='mr').text
    translated_bn = translator.translate(summary, src='en', dest='bn').text

    return ocr_result, summary, translated_hi, translated_mr, translated_bn

# Define the Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="OCR Result"),
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Hindi Translation"),
        gr.Textbox(label="Marathi Translation"),
        gr.Textbox(label="Bengali Translation")
    ],
    title="OCR, Summarization, and Translation",
    description="Upload an image to extract text, summarize it, and translate into Hindi, Marathi, and Bengali."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8000)

