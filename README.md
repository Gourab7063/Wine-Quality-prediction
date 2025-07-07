# 🍷 Wine Quality Prediction

This project is a machine learning-based web application that predicts the quality of wine based on physicochemical features. It leverages multiple classification algorithms and is deployed using Streamlit for an interactive user experience.

## 🚀 Project Overview

Various chemical properties influence wine quality. This model employs supervised learning to categorize wine samples based on their quality. The application allows users to input wine characteristics and receive a predicted quality score.

## 🧠 Algorithms Used

The following machine learning models were trained and evaluated:

1. **Random Forest Classifier** 🌲  
2. **Support Vector Machine (SVM)** 📈  
3. **Multi-layer Perceptron Classifier (MLPC)** 🧠

Each model was evaluated using accuracy, precision, recall, and F1-score to determine the best-performing algorithm.

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy, Scikit-learn**
- **Matplotlib, Seaborn** (for EDA and visualization)
- **Streamlit** (for deployment)

## 📊 Dataset

The dataset used is the [Wine Quality Dataset]([https://archive.ics.uci.edu/ml/datasets/Wine+Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)) from Kaggle. It contains physicochemical attributes of red wine samples along with quality ratings.

## 📈 Model Evaluation

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Random Forest Classifier | 88%      | 89%       | 97%    | 93%      |
| SVM                    | 88%      | 88%       | 93%    | 98%      |
| MLPC                   | 87%      | 91%       | 94%    | 93%      |


## 🌐 Deployment

The model is deployed using **Streamlit**. To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py

## 📁 Project Structure

```
Wine-Quality-Prediction/
│
├── app.py                  # Streamlit app
├── model_rf.pkl            # Trained Random Forest model
├── model_svm.pkl           # Trained SVM model
├── model_mlpc.pkl          # Trained MLPC model
├── winequality.csv         # Dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🙌 Acknowledgements

- Kaggle dataset
- Streamlit for the deployment framework
- Scikit-learn for machine learning models

## 📬 Contact

Created by **Gourab** – feel free to reach out via [LinkedIn](https://www.linkedin.com/) or open an issue for suggestions!

---
