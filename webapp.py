import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn
import pickle
import numpy as np
import pandas as pd
load_model = pickle.load(open('IRIS_model.pkl', 'rb'))

#creating a function for prediction
def iris_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.array(input_data, dtype='float')
    #input_data_as_numpy_array = to_numeric(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = load_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return ('Predicted flower is Setosa')
    elif (prediction[0]== 1):
        return ('Predicted flower is Versicolour')
    else:
        return ('Predicted flower is Virginica')

def main():
    #giving title
    st.title('Iris Flower Classifier ML Web App')
    st.subheader('Developer: Usman Oladapo :sunglasses:')
    st.write("### We need some information to classify flower")

    #getting input from user
    sepal_length = st.text_input("Sepal Length(cm)")
    sepal_width = st.text_input("Sepal Width(cm)")
    petal_length = st.text_input("Petal Length(cm)")
    petal_width = st.text_input("Petal Width(cm)")

    result = ""


    #creating a button for prediction

    if st.button('Predict Flower'):
        result = iris_prediction([sepal_length, sepal_width, petal_length, petal_width])
    st.success(result)

if __name__ == '__main__':
    main()