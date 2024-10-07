# Legal Document Summarize

**Legal Document Summarize** is an all-in-one tool designed to process legal documents from images by extracting text, summarizing the content, and translating the summary into multiple languages. This project combines the power of image preprocessing, Optical Character Recognition (OCR), text summarization using deep learning models, and language translation to provide an efficient solution for legal document analysis.

## Features

### 1. Image Preprocessing
- **Grayscale Conversion**: Converts the input image to grayscale to reduce noise and simplify text extraction.
- **Binarization**: Applies thresholding to the grayscale image to create a binary image, enhancing contrast between text and background.
- **Noise Removal**: Uses techniques like dilation, erosion, and median blurring to clean the image and remove any noise.
- **Deskewing**: Automatically detects and corrects skew in the image, ensuring that text lines are properly aligned for better OCR accuracy.

### 2. Optical Character Recognition (OCR)
- Utilizes `pytesseract`, the Python wrapper for Tesseract OCR, to extract the text from the preprocessed image. This process is capable of handling various image formats and provides a raw text output from the document.

### 3. Text Summarization
- Uses a BERT-based language model (`nlpaueb/legal-bert-base-uncased`) along with `facebook/bart-large-cnn` for summarizing legal content.
- The text is summarized into a concise form, ensuring that only the essential information is retained from large legal documents.

### 4. Multilingual Translation
- Leverages Google Translate to translate the summarized text into multiple Indian languages, including:
  - Hindi
  - Marathi
  - Bengali
- This feature helps in making legal documents more accessible to non-English speakers in India.

## How It Works

1. **Image Upload**: Users can upload images containing legal text.
2. **Preprocessing**: The image goes through a series of preprocessing steps like grayscale conversion, binarization, noise removal, and deskewing to optimize it for OCR.
3. **Text Extraction**: The processed image is passed through Tesseract OCR to extract text.
4. **Text Summarization**: The extracted text is summarized using pre-trained BERT and BART models to provide a concise version of the content.
5. **Translation**: The summarized content is translated into Hindi, Marathi, and Bengali using Google Translate.
6. **Output**: The user receives the raw OCR text, summarized text, and translations in the respective languages.

## Technologies Used

- **Gradio**: To create an easy-to-use user interface for image uploads and display the results.
- **OpenCV**: For image preprocessing tasks like grayscale conversion, binarization, noise removal, and deskewing.
- **Tesseract**: For OCR, enabling the extraction of text from images.
- **Hugging Face Transformers**: For leveraging pre-trained models like BERT and BART for text summarization.
- **Googletrans**: For translating text into multiple Indian languages.
- **Matplotlib**: For visualizing images during preprocessing (optional).

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/legal-document-summarize.git
   cd legal-document-summarize
2. Install the required dependencies:
    bash
    pip install -r requirements.txt
3. Ensure Tesseract is installed on your machine:
    For Windows: Tesseract Installation Guide
    For Linux:
    bash
    sudo apt install tesseract-ocr
4. Launch the Gradio interface:
    python app.py

## Usage
- Open the Gradio web interface in your browser.
- Upload an image of a legal document.
- The application will process the image, extract text, summarize it, and provide translations in Hindi, Marathi, and Bengali.
- You will see the OCR result, summarized text, and translations in the output boxes.

## Future Enhancements
- Add support for more languages.
- Integrate other summarization models for better legal document understanding.
- Improve the image preprocessing pipeline to handle more complex document layouts.

## Contributing
- We welcome contributions! Please fork the repository and submit a pull request.