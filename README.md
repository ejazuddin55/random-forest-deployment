# Random forest Deployment


---

# Random Forest Deployment

This project is a complete end-to-end machine learning pipeline built using **FastAPI** and **Streamlit**, containerized with **Docker** and deployed on **Azure Container Apps**.

### 🚀 Overview

The purpose of this project is to demonstrate how to build, containerize, and deploy a machine learning workflow. It consists of a backend API that trains a Random Forest model and a frontend user interface for interacting with the model.

---

## 🧩 Project Structure

The project is divided into two main components:

### 1. **Backend (FastAPI)**
The backend is responsible for handling API requests, training a Random Forest regression model, and returning performance metrics. It exposes a `/train_model` endpoint to which the frontend sends data and parameters.

- Built with **FastAPI**.
- Accepts CSV files and hyperparameters.
- Trains the model and returns metrics like R² score and MSE.

### 2. **Frontend (Streamlit)**
The frontend provides a simple UI for users to upload their datasets, adjust model parameters (like `n_estimators`, `criterion`, etc.), and visualize the training results.

- Built with **Streamlit**.
- Users can interact with the model via sliders, dropdowns, and file uploaders.
- Sends user input and data to the backend for model training.

---

## 🐳 Docker & Docker Compose

Both the backend and frontend are containerized separately using Docker. Docker Compose is used to orchestrate both services locally.

- Backend Dockerfile sets up FastAPI.
- Frontend Dockerfile sets up Streamlit.
- **docker-compose.yml** defines how the two services communicate via a network.

---

## ☁️ Deployment

The project is deployed using **Azure Container Apps**:

- Backend image is pushed to a private Azure Container Registry (ACR).
- The backend container is hosted via Azure Container Apps.
- The frontend communicates directly with the deployed backend via its Azure URL.

---

## 🛠️ Tech Stack

- **Python**
- **FastAPI**
- **Streamlit**
- **Scikit-learn**
- **Pandas**
- **Docker**
- **Azure Container Apps**

---

## 🎯 Goal

The goal is to make it easy for users to:
- Upload datasets.
- Train and evaluate Random Forest models.
- View results on a simple web interface.
- Learn how to containerize and deploy ML models using cloud services.

