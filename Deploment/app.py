import streamlit as st
import numpy as np
import pickle
import pre_process

vect = pickle.load(open('bow.pkl', 'rb')) # Loading Count Vectorizor
lgr = pickle.load(open('lgr.sav', 'rb')) # Loading Model

st.title('Sexual Harassment')
st.image('https://www.talkingnibs.com/wp-content/uploads/2018/03/MeToo-2.jpg', width = 375)
query = st.text_input('Enter the event description:')
if query == None or query == '':
    st.markdown('**Enter a text to get result...**')
else:
    query = pre_process.preprocess_text([query])
    query = vect.transform(query)

    pred = lgr.predict(query)[0] # Predicting Classes

    st.markdown("""<style>.big-font {font-size:20px !important;}</style>""", unsafe_allow_html=True)

    st.markdown('<p class="big-font">Possible Act:</p>', unsafe_allow_html=True)

    res = []
    if pred[0] != 0: # Simple if-else for simplified understanding of Results
        res.append('Commenting')

    if pred[1] != 0:
        res.append('Ogling')

    if pred[2] != 0:
        res.append('Touching')

    if sum(pred) == 0: # Adding A Neutral Class if prediction does belong to any classes and its not effecting/related to training/result of model
        res.append('Neutral')

    st.markdown(res)
