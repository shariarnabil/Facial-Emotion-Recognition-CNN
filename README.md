## Facial Emotions Detection

Welcome to the Human Facial Emotions Detection project! This application uses a Convolutional Neural Network (CNN) implemented with TensorFlow/Keras to detect and classify human facial emotions from images. The system is built as a Flask web app, allowing users to upload images and receive predictions on emotions such as angry, disgust, fear, happy, neutral, sad, and surprise.

## Overview

This project leverages deep learning to analyze facial expressions in grayscale images (48x48 pixels) and predict the corresponding emotion. The trained model is integrated into a web interface where users can upload images and view the predicted emotion along with a confidence score.

- **Last Updated**: August 15, 2025, 08:47 AM +06


## Features

- Upload images to detect facial emotions.
- Real-time prediction with confidence percentage.
- Modern, responsive web interface with a loading spinner.
- Support for multiple pre-trained model files.
- Error handling for missing models or invalid images.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.8+**
- Required Python packages:
  - `flask`
  - `tensorflow`
  - `numpy`
  - `pillow`

Install the dependencies using pip:

```bash
pip install flask tensorflow numpy pillow