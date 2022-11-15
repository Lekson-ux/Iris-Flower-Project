import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
import sklearn
import streamlit
import numpy as np
import pandas as pd
#loaded_model = pickle.load(open('C:/Users/USER/Documents/Streamlit ML APPS/Diabetes Project/Saved_model.pkl', 'rb'))
load_model = pickle.load(open('IRIS_model.pkl', 'rb'))
input_data = (7.0, 3.2, 4.7, 1.4)

# changing the input_data to numpy array
input_data = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data.reshape(1,-1)

prediction = load_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('Predicted flower is Setosa')
elif (prediction[0]== 1):
    print('Predicted flower is Versicolour')
else:
    print('Predicted flower is Virginica')