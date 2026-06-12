# 🧠 Brain Tumor Detection

A deep-learning web application that classifies brain MRI scans into four categories — **glioma, meningioma, pituitary tumor, and no tumor** — using a Convolutional Neural Network, served through a Flask web app with real-time image upload.

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=flat&logo=mongodb&logoColor=white)

## ✨ Features

- **Four-class classification** of MRI scans (glioma · meningioma · pituitary · no tumor) with a CNN built in TensorFlow/Keras.
- **Preprocessing and data augmentation** to improve accuracy and reduce overfitting.
- **Flask web app** with image upload for real-time predictions.
- **MongoDB integration** for storing user data, prediction history, and appointments.

## 🛠️ Tech Stack

**ML:** Python, TensorFlow, Keras, NumPy, OpenCV/Pillow
**Web:** Flask, HTML/CSS, JavaScript
**Database:** MongoDB

## 📊 Model

- Architecture: CNN <!-- TODO: note layers / or "transfer learning with <base model>" if used -->
- Dataset: <!-- TODO: link the MRI dataset, e.g. the Kaggle Brain Tumor MRI dataset -->
- Performance: <!-- TODO: add your real validation accuracy / confusion matrix -->

## 🚀 Getting Started

```bash
# 1. Clone
git clone https://github.com/fenilramani4007/Brain_Tumor_Detection.git
cd Brain_Tumor_Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure MongoDB is running locally (or set your connection string)

# 4. Run the app
python app.py   # <!-- TODO: confirm entry-point filename -->
```

Then open `http://127.0.0.1:5000` and upload an MRI image.

## 📁 Project Structure

```
Brain_Tumor_Detection/
├── app.py                # Flask entry point   <!-- TODO: confirm -->
├── model/                # trained model + notebook
├── static/ , templates/  # web UI
├── requirements.txt
└── README.md
```

## ⚠️ Disclaimer

This project is for **educational and research purposes only** and is **not a medical device**. It must not be used for clinical diagnosis.

## 📬 Contact

Fenil Ramani — [LinkedIn](https://linkedin.com/in/fenil-ramani-dev)
