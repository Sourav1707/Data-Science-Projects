import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from sklearn.preprocessing import StandardScaler




def welcome():
    return "Welcome All"

def predict_heart_failure(age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time,model_no):
    scaler = StandardScaler()

    

    if model_no ==  1 :
        pickle_in = open("logistic.pkl","rb")
        lg_model=pickle.load(pickle_in)
        prediction=lg_model.predict(scaler.fit_transform([[age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time]]))
    elif model_no ==2:
        pickle_in = open("knn.pkl","rb")
        knn_model=pickle.load(pickle_in)
        prediction=knn_model.predict(scaler.fit_transform([[age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time]]))
    elif model_no == 3:
        pickle_in = open("rf.pkl","rb")
        rf_model=pickle.load(pickle_in)
        prediction=rf_model.predict(scaler.fit_transform([[age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time]]))
    elif model_no == 4:
        pickle_in = open("dt.pkl","rb")
        dt_model=pickle.load(pickle_in)
        prediction=dt_model.predict(scaler.fit_transform([[age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time]]))
    print(prediction)
    predict = ""
    if prediction:
        predict = "True"
    else:
        predict = "False"
    return predict

def main():
    st.title("Heart Failure Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Heart Failure Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input("age","Type Here")
    creatinine_phosphokinase = st.text_input("creatinine_phosphokinase","Type Here")
    ejection_fraction = st.text_input("ejection_fraction","Type Here")
    serum_creatinine = st.text_input("serum_creatinine","Type Here")
    serum_sodium = st.text_input("serum_sodium","Type Here")
    time = st.text_input("time","Type Here")
    # result=""

    
    if st.button("Logistic"):
        # result=predict_heart_failure(age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time)
        st.success('The output is {}'.format(predict_heart_failure(age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time,1)))
    if st.button("KNN"):
        st.success('The output is {}'.format(predict_heart_failure(age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time,2)))
    if st.button("Random Forest"):
        st.success('The output is {}'.format(predict_heart_failure(age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time,3)))
    if st.button("Decision Tree"):
        st.success('The output is {}'.format(predict_heart_failure(age,creatinine_phosphokinase,ejection_fraction,serum_creatinine,serum_sodium,time,4)))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()