#importing necessary library
import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rfc=joblib.load('rfc_model.pkl')
clf=joblib.load('clf_model.pkl')
mlpc=joblib.load('mlpc_model.pkl')
wine=pd.read_csv("winequality-red.csv",sep=',')
X=wine.drop('quality',axis=1)
y=wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

st.title("Welcome to my Wine Quality Prediction App")
st.write('''Enter the values of the following features to predict the wine quality like''')

components=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
for i in components:
    st.write(i)
# import streamlit as st
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

# Input features
X_input =[[7.3,0.650,0.00,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0]]

X_scaled = sc.transform(X_input)

# Predictions
pred_rf = rfc.predict(X_scaled)[0]
pred_clf = clf.predict(X_scaled)[0]
pred_mlp = mlpc.predict(X_scaled)[0]

# Display results
st.subheader("Wine Quality Predictions")
st.write(f"🍷 Random Forest Prediction: **{pred_rf}**")
st.write(f"🔍 Classifier Prediction: **{pred_clf}**")
st.write(f"🧠 Neural Network Prediction: **{pred_mlp}**")
