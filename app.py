# -*- coding: utf-8 -*-
"""
Created on 3/20/2023

@author: Temesgen
"""
from PIL import Image
import numpy as np 
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models

diabetes_model = pickle.load(open('E:/final year project/Multiple Disease Prediction System/saved models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('E:/final year project/Multiple Disease Prediction System/saved models/heart_disease_model.sav','rb'))
# model = load_model("keras_Model.h5")



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Skin disease prediction'
                           ],
                          icons=['activity','heart','skin'],
                                                 
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
        # if Age=='':
        #     st.error('please fill it')
        
        #     st.stop()
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic \n\n Management:\n\n 1)abc \n\n2)abcd'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
              
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        

# def fruit_disease(img, weights_file):
#     # Load the model
#     model = tf.keras.saving.load_model(weights_file)

#     # Create the array of the right shape to feed into the keras model
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#     image = img
#     #image sizing
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.ANTIALIAS)

#     #turn the image into a numpy array
#     image_array = np.asarray(image)
#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # run the inference
#     prediction = model.predict(data)
#     return np.argmax(prediction) # return position of the highest probability

    
if (selected=="Skin disease prediction"):
    st.title('Skin disease prediction')
    
    # Function to Read and Manupilate Images
    def load_image(img):
        im = Image.open(img)
        
        image = np.array(im)
        return image  
   
# Uploading the File to the Page
    uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png','jpeg'])

    if uploadFile is not None:
        # Perform your Manupilations (In my Case applying Filters)
        img = load_image(uploadFile)
        st.image(img,width=200)
        # label==1 
        # # label=fruit_disease(img,'keras_Model.h5')
        # # 'Anth', 'Healthy', 'Not mango', 'Powdery', 'White Scale'
        # if label==0:
        #     st.write('Anthranose')
        # elif label==1:
        #     st.write('Healthy')
        # elif label==2:
        #     st.write('Not mango')
        # elif label==3:
        #     st.write('Powdery')
        # else:
        #     st.write('white Scale')
        st.write("Image Uploaded Successfully")
    else:
        st.write("Make sure you image is in JPG/PNG/JPEG Format.")
    
    if st.button('predict'):
        st.header('Comming soon!')
        st.subheader('skin disease and treatments')
             
    # st.success(diab_diagnosis)
