# Air Quality Forecasting System

This project implements a serverless inference engine on **Azure Functions** and a local training pipeline for predicting air quality metrics, using Deep Learning (LSTM, Bi-LSTM, and Attn-LSTM) and Machine Learning (Random Forest) models.

## Architecture

* **Inference:** A serverless backend deployed on Azure Functions (HTTP requests)
* **Training:** Local Python script using TensorFlow/Keras and Scikit-learn.

## Supported Models

The system allows hot-swapping between the following architectures:
* **LSTM:** Standard Long Short-Term Memory.
* **Bi-LSTM:** Bidirectional LSTM.
* **Attn-LSTM:** LSTM with attention mechanism.
* **RF:** Random Forest Regressor.