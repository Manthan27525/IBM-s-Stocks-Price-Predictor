import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


st.title("IBM Stocks Prediction")

open = st.number_input('Open Price')
high = st.number_input('High Price')
low = st.number_input('Low Price')
volume = np.log1p(st.number_input('Volume'))

if st.button('Predict Close Price'):
    query = np.array([[open,high,low,volume]])
    query=scaler.transform(query)
    st.text("The predicted close price is :")
    st.title(f"{model.predict(query)[0]:.2f}")