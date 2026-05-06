# 🌱 CropCare: Plant Disease Detector

CropCare is a lightweight machine learning application that detects crop diseases from leaf images. It uses a **MobileNetV2** model trained on the PlantVillage dataset to classify images and estimate disease severity. The application is built using **Streamlit** and supports features like severity estimation, multilingual support (Hindi), and voice output.

## Features
- **Disease Classification:** Identifies common plant diseases from leaf images.
- **Severity Estimation:** Calculates the percentage of the affected area on the leaf using OpenCV.
- **Multilingual Support:** Displays disease names and recommendations in both English and Hindi.
- **Voice Output:** Speaks the diagnosis and severity using Text-To-Speech.

## Getting Started

### 1. Prerequisites
Make sure you have Python 3.9+ installed.

### 2. Installation
Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Dataset Setup
Download the [PlantVillage Dataset](https://www.kaggle.com/mohitsingh1804/plantvillage) from Kaggle.
Extract the images and place them in the `dataset/` directory inside this project folder. Your structure should look like this:

```
dataset/
  Tomato_Early_Blight/
  Tomato_Late_Blight/
  Tomato_Healthy/
  Potato_Late_Blight/
  ...
```

*(Note: We recommend starting with a smaller subset like Tomato and Potato classes for faster training.)*

### 4. Running the Application (Coming Soon)
Once the model is trained, you will be able to run the Streamlit app using:

```bash
streamlit run src/app.py
```
