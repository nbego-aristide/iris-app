import streamlit as st
import numpy as np
import joblib
import pandas as pd


sepal_long = st.slider("Entrez la longueur du sepal", 0.0, 10.0)
sepal_larg = st.slider("Entrez la largeur du sepal", 0.0, 10.0)
petal_long = st.slider("Entrez la longueur du petal", 0.0, 10.0)
petal_larg = st.slider("Entrez la largeur du petal", 0.0, 10.0)


if st.button("Prediction sur la fleur", type="primary"): 
    model= joblib.load("my_model.pkl")
    scaler= joblib.load("my_scaler.pkl")
    
    f = np.array([[sepal_long, sepal_larg, petal_long, petal_larg]])
    X = pd.DataFrame(f,columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])

    
    prediction = model.predict(X)
    
    response = prediction[0]

 
    st.write("fleur est: ", response)
   
