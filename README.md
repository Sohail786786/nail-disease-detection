# Nail Disease Detection 🩺

Detecting nail diseases using deep learning + Flask web app.

---

## 🚀 Project Overview

This project uses a Convolutional Neural Network (EfficientNetB0) to classify images of nails into six categories:

- Acral Lentiginous Melanoma  
- Healthy Nail  
- Onychogryphosis  
- Blue Finger  
- Clubbing  
- Pitting  

Users can upload a nail image via a web interface, and the app returns the predicted disease + confidence percentage.

---

## 🛠️ Technologies Used

- **Python**  
- **TensorFlow / Keras**  
- **Flask**  
- **HTML / CSS / JavaScript**  
- **NumPy, Matplotlib, scikit-learn**  

---

## 📂 Folder Structure

nail-disease-detection/
│
├── model/ # Saved model file (ignored in repo)
├── Dataset/ # Dataset (ignored in repo)
├── static/
│ ├── images/ # Any UI images
│ └── uploads/ # Where user uploads are saved (ignored)
├── templates/
│ ├── index.html
│ ├── about.html
│ ├── nailhome.html
│ └── nailpred.html
├── app.py # Flask backend
├── train_model.py # Script for training
├── test_model.py # Script for evaluating model
├── requirements.txt # Dependency list
└── README.md # This file


> **Note:** The `model/`, `Dataset/`, and `static/uploads/` directories are excluded from the repo (via `.gitignore`) because they contain large files or user uploads.

---

## 🧪 Setup & Run Instructions

1. **Clone your repo**
   ```bash
   git clone https://github.com/Sohail786786/nail-disease-detection.git
   cd nail-disease-detection
2. Install dependencies
  pip install -r requirements.txt
3.Run the web app
  python app.py
4.Open in browser
  Go to http://127.0.0.1:5000/ to use the app.

🧠 Model Details

Model architecture: EfficientNetB0 (no pre-trained weights)

Input image size: 224 × 224 × 3

Output classes: 6 (listed above)

Class balancing: compute_class_weight used to mitigate dataset imbalance

Callbacks used: EarlyStopping, ReduceLROnPlateau

<img width="1891" height="1036" alt="Screenshot 2025-10-05 191204" src="https://github.com/user-attachments/assets/ee3a0d1d-8ec7-4900-ac06-e6d3cd16198c" />

<img width="1890" height="992" alt="Screenshot 2025-10-05 191228" src="https://github.com/user-attachments/assets/e999ea7a-515b-49f8-b80f-370b8118cb01" />

<img width="1892" height="974" alt="Screenshot 2025-10-05 191243" src="https://github.com/user-attachments/assets/6785c127-fda8-4fe9-b85e-c63edd86aa34" />

<img width="1899" height="897" alt="Screenshot 2025-10-05 191256" src="https://github.com/user-attachments/assets/fc75e845-ba78-459d-8c79-53c0b5905009" />

<img width="1878" height="985" alt="Screenshot 2025-10-05 191312" src="https://github.com/user-attachments/assets/7bc7d28e-3e42-4775-a370-a0896cbf49c2" />








