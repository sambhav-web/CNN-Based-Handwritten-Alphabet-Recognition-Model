import streamlit as st
from PIL import Image
import numpy as np
import joblib
with open('model.pkl','rb') as f:
  model=joblib.load(f)
st.title('Handwritten Alphabet Recognition')
st.divider()
st.info('This model predicts the black and white image as the alphabets.')
img=st.file_uploader('Upload Image',help='Enter Black and White Image')
background=st.selectbox('Choose Background Color :',['Blackish','Whitish'])
def get_alphabet_letter(prediction_num):
    """
    Converts a numerical class label (0 to 25) into the corresponding letter (A to Z).
    """
    if 0 <= prediction_num <= 25:
        return chr(65 + int(prediction_num))
def preprocess_img(image,color):
  img=Image.open(image).convert('L')
  img=img.resize((28,28))
  img=np.array(img)
  if color=='Whitish':
    img=255-img
  img=img/255
  img=img.reshape(1,28,28,1)
  pred=model.predict(img)
  pred=np.argmax(pred)
  return img,get_alphabet_letter(pred)
if img:
  image,pred=preprocess_img(img,background)
  st.image(image)
  st.write(f'Prediction: {pred}')