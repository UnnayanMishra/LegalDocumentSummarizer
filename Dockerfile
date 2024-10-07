# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system-level dependencies for Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev

# Install Python dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the web server (Gradio will use this)
EXPOSE 8000

# Ensure Tesseract is in the PATH (if needed)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
ENV PATH="$PATH:/usr/local/bin"

# Make the start script executable (optional if not already executable)
RUN chmod +x start.sh

# Start the application (assuming your app entry point is in `start.sh`)
CMD ["bash", "start.sh"]
