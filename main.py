import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
tf.compat.v1.reset_default_graph()
 
import cv2
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10 # type: ignore

import streamlit as st
from PIL import Image

st.markdown("""
<style>
.header {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
    vertical-align: top;
}
</style>
<div class="header">CIFAR-10 Image Classification</div>
""", unsafe_allow_html=True)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0) 

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#NORMALIZATION
# Convert pixel values data type to float32
X_train = X_train.astype('float32')
# X_test  = X_test.astype('float32')
# X_valid = X_valid.astype('float32')

# Calculate the mean and standard deviation of the training images
mean = np.mean(X_train)
std  = np.std(X_train)

from tensorflow.keras.models import load_model # type: ignore
model = load_model('saved_models/model_20_epochs.keras')


uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:

    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32,32))
    image = (image-mean)/(std+1e-7)
    image = image.reshape((1, 32, 32, 3))
    
    prediction = model.predict(image)
    predicted_class = prediction.argmax()
    
    st.write('Predicted class: ', class_names[predicted_class])
    
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)   
    
else:
    st.write("File Not Uploaded")