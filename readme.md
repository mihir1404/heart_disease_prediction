
# 💓 Heart Disease Prediction App

This repository contains a web application built with **Streamlit** and a trained **Artificial Neural Network (ANN)** model to predict the risk of heart disease based on user-provided medical parameters.

---

## 🌟 Overview

The app uses a pre-trained ANN model to classify whether a person is at risk of heart disease based on the following inputs:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting ECG Result
- Max Heart Rate Achieved
- Exercise Induced Angina
- ST Depression (Oldpeak)
- Slope of Peak Exercise ST Segment
- Number of Major Vessels Colored
- Thalassemia

Once the values are entered, the model predicts:

✅ **No risk of heart disease**  
❌ **Risk of heart disease**

---

## 🖥️ Live Demo (Optional)

> You can run this app locally (instructions below), or host it on [Streamlit Community Cloud](https://streamlit.io/cloud) for public access.

---

## 📁 Project Structure

```
├── app.py                               # Streamlit app to run the prediction
├── model_train.py                       # Model training script using Keras
├── scaler.pkl                           # Scaler used to normalize input features
├── heart_ann_model.h5                   # Trained ANN model
├── Dataset--Heart-Disease-Prediction-using-ANN.csv  # Heart disease dataset
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## 🧠 Model Details

- Built using **Keras Sequential API**
- Architecture:
  - Input layer: 13 features
  - Hidden layers: 2 Dense layers (32 units, ReLU)
  - Output layer: 1 unit (Sigmoid)
- Optimizer: `Adam`
- Loss Function: `Binary Crossentropy`
- Training Epochs: 100
- Batch Size: 8
- Evaluation Metric: Accuracy
- Scaler: StandardScaler saved via `joblib`

---

## 🏃‍♂️ How to Run the App Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create and Activate Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Dataset

- **File**: `Dataset--Heart-Disease-Prediction-using-ANN.csv`
- **Shape**: 303 samples × 14 columns (13 features + 1 target)
- **Target Column**: `target` (0 = No Disease, 1 = Disease)
- **Source**: Public dataset (similar to UCI Heart Disease dataset)

---

## 🔁 Retrain the Model

If you wish to retrain the ANN model:

```bash
python model_train.py
```

This will:
- Load the dataset
- Normalize it using `StandardScaler`
- Train an ANN with 2 hidden layers
- Save:
  - `heart_ann_model.h5` (the trained model)
  - `scaler.pkl` (fitted scaler for preprocessing)

---

## 📦 Requirements

See `requirements.txt` for the full list. Key packages:

```
streamlit==1.35.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
scikit-learn==1.4.2
joblib==1.4.2
tensorflow==2.16.1
keras==3.3.3
```

---

## 🙋 Author

**Mihir**  
🎓 MTech ICT – Machine Learning | DA-IICT  
📫 GitHub: [@your-username](https://github.com/your-username)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
