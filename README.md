# 🎓 College Student Placement Prediction

This is a complete end-to-end Machine Learning project that predicts whether a student is likely to be placed based on academic and personal parameters.

It includes a full pipeline with **EDA, Model Building, Flask API, Streamlit UI, and Docker deployment**.

---

## 📊 Project Overview

The goal is to help:
- Students understand their placement chances  
- Institutions analyze key placement factors  

The model uses historical data to identify patterns and predict outcomes.

---

## ✨ Key Features

- 📊 Exploratory Data Analysis (EDA)  
- 🤖 Naive Bayes ML Model + PCA  
- 🌐 Flask Backend API  
- 🖥️ Streamlit Interactive UI  
- 🐳 Dockerized Deployment  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Backend:** Flask  
- **Frontend:** Streamlit  
- **DevOps:** Docker, Shell Scripting  

---

## 📂 Project Structure

```text
├── Docker_Files/
├── ds-project-1.ipynb
├── backend.py
├── frontend.py
├── config.py
├── college_student_placement_dataset.csv
├── likelihood_distribution_params.pkl
├── eigen_vectors.npy
├── requirementstxt
└── start-backend-frontend.sh
```
---

## 🚀 Installation & Setup

### 🔧 Prerequisites

Make sure you have installed:

- Python 3.10+  
- pip  
- Git  
- Docker (optional)  

---

### 📥 Clone the Repository

```bash
git clone https://github.com/AviralStack/Ds-Project-1.git
cd Ds-Project-1

```
### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```
### ▶️ Running the Project
#### 🟢 Method 1: Using Shell Script (Recommended)
```bash
bash start-backend-frontend.sh
```
### 🌐 Access the Application
#### Frontend: http://localhost:8501
#### Backend API: http://localhost:5000




