# 🚗 Car Insurance Claim Prediction System

A machine learning–powered web application that predicts the **probability of an insurance claim** based on customer, vehicle, location, and safety-related features. The project demonstrates an **end‑to‑end ML workflow** — from data preprocessing and model training to deployment using **Streamlit**.

---

## 📌 Project Overview

Insurance companies need to accurately assess risk to price policies and manage claims efficiently. This application helps estimate the **likelihood of a car insurance claim** for a given customer profile through a **complete machine‑learning system** that includes data preprocessing, feature engineering, model training, evaluation, and deployment. While a **Random Forest classifier** is used as the final predictive model, the core focus of the project is building a **reliable, end‑to‑end prediction system**, not just training a single algorithm.

### Key Highlights

* End‑to‑end ML pipeline (EDA → training → evaluation → deployment)
* Handles **class imbalance** using `class_weight="balanced"`
* Interactive **Streamlit** web interface
* Supports **single prediction** (and optional batch prediction)
* Clean, modular, and portfolio‑ready structure

---

## 🧠 Machine Learning & System Design Details

* **Model (Final Estimator)**: RandomForestClassifier
* **Pipeline Components**: Data preprocessing, feature alignment, model inference, probability calibration, and UI‑level validation
* **Problem Type**: Binary Classification (Claim / No Claim)
* **Evaluation Metric**: ROC‑AUC (primary), Precision, Recall
* **Class Imbalance Handling**: `class_weight='balanced'`

### Hyperparameter Tuning Note

Full `GridSearchCV` was attempted; however, due to **system memory constraints on Windows**, large parallel grid search runs were limited. Final hyperparameters were selected using **partial tuning and validation performance**, which is a common real‑world ML engineering trade‑off.

---

## 🧾 Features Used

### Policy & Customer

* Policy tenure (months)
* Policyholder age

### Vehicle

* Car age
* Segment
* Fuel type
* Engine displacement
* Max power
* Max torque

### Safety

* Airbags
* NCAP rating
* ESC
* Brake assist
* Parking sensors / camera

### Location

* Area cluster
* Population density

---

## 🖥️ Web Application (Streamlit)

### Pages

* **Single Prediction**: Predict claim probability for one customer
* **Batch Prediction** (optional): Upload CSV and get predictions
* **About**: Project overview and usage info

### Output

* Claim probability (%)
* No‑claim probability (%)
* Risk category (Low / Medium / High)

---

## 🗂️ Project Structure

```
car-insurance-claim-prediction/
│
├── app.py                     # Streamlit application
├── models/
│   └── best_model.pkl           # Trained model
├── data/                      # Raw / processed data (optional)
├── src/                       # EDA & training notebooks (optional)
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Pooja-p18/car-insurance-claim-prediction.git
cd car-insurance-claim-prediction
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

---

## 📊 Model Evaluation (Summary)

* The model produces **stable and interpretable probabilities**
* Predictions respond logically to risk‑increasing and risk‑reducing features
* Suitable for **demonstration and educational purposes**

---

## 🚀 Deployment

This application is deployment‑ready and can be hosted using:

* **Streamlit Community Cloud**
* Any cloud VM supporting Python

Before deployment:

* Ensure `requirements.txt` contains only required libraries
* Ensure model path is correct (`models/best_model.pkl`)

---

## 🔮 Future Improvements

* Advanced feature engineering
* SHAP‑based model explainability
* Better hyperparameter tuning with higher‑resource environment
* User authentication
* API version (FastAPI)

---

## 👩‍💻 Author

**Pooja Parashuram Bajantri**
Computer Science Engineer | Aspiring Data Scientist / Data Analyst

---

## 📜 License

This project is for **educational and portfolio purposes**.

---

⭐ If you found this project useful, consider giving it a star!
