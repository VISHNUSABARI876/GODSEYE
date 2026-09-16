# AI-Image-Video-Detector

A production-style Deep Learning powered Web Application to detect whether an uploaded image or video is real or AI-generated (Deepfake / Synthetic Media).

---

## 🏗️ Architecture Overview

The system uses a clean, modular architecture:

* **Frontend**: React 18, Vite, Javascript, Axios API Client, Responsive Dark Mode CSS Design System.
* **Backend**: Python 3.10+, Flask, Flask-CORS Blueprint Architecture.
* **Database**: SQLite with SQLAlchemy ORM for user accounts & detection history storage.
* **ML / Computer Vision Stack**: PyTorch, OpenCV, Pillow, NumPy (modular pipeline to be integrated in Phase 2).
* **Authentication**: JWT token-based authentication.

---

## 📁 Project Folder Structure

```
AI-Image-Video-Detector/
│
├── frontend/                  # React + Vite Frontend Application
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Page views & layouts
│   │   ├── services/          # Axios API communication services
│   │   ├── hooks/             # Custom React hooks
│   │   ├── utils/             # Helper functions
│   │   ├── App.jsx            # Main React component (Health Dashboard)
│   │   ├── main.jsx           # React app entry point
│   │   └── index.css          # Dark Mode glassmorphism styling
│   ├── package.json           # Frontend dependencies
│   ├── vite.config.js         # Vite configuration with backend proxy
│   └── .env.example           # Frontend environment variable template
│
├── backend/                   # Flask REST API Application
│   ├── app.py                 # Flask app factory, CORS, and health check
│   ├── config.py              # Environment configuration & settings
│   ├── requirements.txt       # Python package dependencies
│   ├── .env.example           # Backend environment variable template
│   │
│   ├── routes/                # Modular Flask Blueprints
│   │   ├── auth_routes.py     # User registration & JWT authentication
│   │   ├── detection_routes.py# Image & Video AI detection endpoints
│   │   └── history_routes.py  # Analysis log & history management
│   │
│   ├── models/                # SQLAlchemy database models
│   │   ├── database.py        # DB instance & initialization
│   │   ├── user.py            # User account schema
│   │   └── detection.py       # Detection audit log schema
│   │
│   ├── services/              # Business logic & detection handlers
│   │   ├── image_detector.py  # Image classifier wrapper
│   │   ├── video_detector.py  # Video frame aggregator
│   │   ├── frame_extractor.py # OpenCV video frame processing
│   │   └── preprocessing.py   # Tensor transform & scaling
│   │
│   └── ml/                    # Machine Learning pipeline
│       ├── models/            # PyTorch neural network architectures
│       ├── datasets/          # Custom dataset loaders
│       ├── training/          # Model training & validation scripts
│       └── inference/         # Forward pass & evaluation
│
├── data/                      # Dataset organization
│   ├── raw/                   # Original dataset files
│   ├── processed/             # Cleaned & augmented images
│   ├── train/                 # Training set split
│   ├── validation/            # Validation set split
│   └── test/                  # Evaluation test set
│
├── scripts/                   # Utility maintenance scripts
├── uploads/                   # Temporary directory for uploaded media files
├── README.md                  # Project documentation
└── .gitignore                 # Version control exclusion list
```

---

## 🚀 Quick Start Guide

### Prerequisites
* **Python**: `3.10` or higher
* **Node.js**: `v18.0.0` or higher (with `npm` or `yarn`)

---

### 1️⃣ Setting Up & Running the Backend (Flask)

1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```

2. (Optional but recommended) Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create environment configuration file:
   ```bash
   cp .env.example .env
   ```

5. Launch the Flask development server:
   ```bash
   python app.py
   ```
   *The backend REST API will run on `http://127.0.0.1:5000`.*

---

### 2️⃣ Setting Up & Running the Frontend (React + Vite)

1. Open a new terminal window and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Create frontend environment configuration file:
   ```bash
   cp .env.example .env
   ```

4. Launch the Vite development server:
   ```bash
   npm run dev
   ```
   *The React application will open on `http://localhost:3000` (or `http://localhost:5173`).*

---

## ☁️ Render Deployment Configuration

When deploying the backend web service on Render, ensure the following configuration is used (defined in `render.yaml`):

* **Root Directory**: `backend`
* **Build Command**:
  ```bash
  pip install --upgrade pip && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.0 torchvision==0.18.0 && pip install -r requirements.txt && python -c "import torch; import torchvision; print('TORCH:', torch.__version__); print('TORCHVISION:', torchvision.__version__)"
  ```
* **Start Command**:
  ```bash
  gunicorn app:app
  ```
> **Note**: Because the Root Directory is set to `backend`, the Start Command is `gunicorn app:app` without any extra sub-directory flags.


