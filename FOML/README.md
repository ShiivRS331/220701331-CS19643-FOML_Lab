# P.R.I.S.M - P
## Fingerprint-Based Blood Group Prediction System

## Description

This project presents a non-invasive approach to predicting an individual's blood group using fingerprint analysis. By leveraging Convolutional Neural Networks (CNNs) and integrating blockchain technology, the system ensures accurate predictions while maintaining data integrity and security. The user-friendly interface, built with Streamlit, allows users to upload fingerprint images and receive instant blood group predictions along with compatibility information.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

## Features

- **Fingerprint Image Upload**: Users can upload fingerprint images in formats like JPEG, PNG, or BMP.
- **Blood Group Prediction**: Utilizes a trained CNN model to predict the blood group and Rh factor.
- **Blood Compatibility Checker**: Provides information on compatible blood types for transfusions.
- **Blockchain Integration**: Ensures data integrity by recording transactions securely.
- **User Feedback Mechanism**: Allows users to provide feedback, aiding in continuous learning and improvement.

## System Architecture

The system is structured into three primary layers:

1. **Frontend Layer**: Built with Streamlit, this layer handles user interactions, including image uploads and displaying results.
2. **Backend Layer**: Processes the uploaded images, performs preprocessing, and communicates with the model layer.
3. **Model Layer**: Contains the CNN model trained to classify fingerprint images into specific blood groups.

### Workflow

1. User uploads a fingerprint image via the web interface.
2. The image is sent to the backend for preprocessing.
3. The preprocessed image is passed to the CNN model for prediction.
4. The predicted blood group and compatibility information are returned to the frontend and displayed to the user.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ShiivRS331/220701331-CS19643-FOML_Lab.git
   cd 220701331-CS19643-FOML_Lab/FOML
