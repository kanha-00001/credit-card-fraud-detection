
# 🧠 Credit Card Fraud Detection using Machine Learning

This project implements a machine learning pipeline to detect fraudulent credit card transactions. It involves data preprocessing, handling class imbalance with SMOTE, training a neural network (ANN), and visualizing results.

---

## 📁 Project Structure

```
CC_ML_Project/
├── CC_ML_Project.ipynb        # Main Jupyter notebook
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
└── data/                      # Folder to store data (not included in repo)
```

---

## 🧰 Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- TensorFlow / Keras
- Matplotlib, Seaborn
- Jupyter Notebook
- VS Code (for development)

---

## 🚀 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

---

## 🧪 Features & Methodology

- **Data Cleaning**: Handling missing values and preprocessing features.
- **Train-Test Split**: 80/20 split using `train_test_split`.
- **Class Imbalance Handling**: SMOTE oversampling on minority class.
- **Model**: Artificial Neural Network (ANN) using TensorFlow/Keras.
- **Evaluation**: Accuracy, Precision, Recall, F1 Score via `classification_report`.
- **Visualization**: Class distribution before/after SMOTE using seaborn/matplotlib.

---

## 📊 Results

- Balanced dataset via SMOTE
- Improved recall for minority class (fraudulent transactions)
- Graphs showing class distribution before and after SMOTE

---
