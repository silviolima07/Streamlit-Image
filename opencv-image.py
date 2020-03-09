# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions 

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2
from PIL import Image
from bokeh.models.widgets import Div
from matplotlib import pyplot as plt
from collections import Counter

from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT, DEFAULT_NEIGHBORS_THRESHOLD
from constant import CLASSES, COLORS

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

def linkedin():
    if st.button('Linkedin'):
            js = "window.open('https://www.linkedin.com/in/silviocesarlima/')"  # New tab or window
            #js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)


def main():
    global image, img_file_buffer,flag_canny
    
    st.sidebar.title("Escolha uma opção")
    app_mode = st.sidebar.selectbox(" ",["Classificar","Detectar Olhos","Canny","Gray","About", "Silvio Lima"])
    if  app_mode == "Classificar":
        classificar()
    elif app_mode == "Detectar Olhos":    
        detectar_olhos()
    elif app_mode == "Canny":    
        canny()
    elif app_mode == "Gray":    
        gray()
    elif app_mode == "About":    
        about()
    elif app_mode == "Silvio Lima":
        linkedin()
        
            



@st.cache
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections

def about():
    img = cv2.imread("images/streamlit.png")
    st.sidebar.title("About")
    st.sidebar.info(
    "Web app construido com a lib Streamlit.")
    st.sidebar.info(
    "Streamlit é uma biblioteca do Python que torna fácil construir apps web.")
    st.sidebar.info(
    "Semelhante ao Shiny usado com R.")
    st.sidebar.info(
    "Github: https://github.com/streamlit/streamlit")
    st.sidebar.info(
    "Streamlit: https://streamlit.io")
    st.sidebar.info(
    "Referência1: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/")
    st.sidebar.info(
    "Referẽncia2: https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5")
    st.write("")
    st.image(img, use_column_width=True)

@st.cache
def annotate_image(
    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    # loop over the detections
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0), 2  # Black text
            )
    return image, labels

def classificar():

    st.markdown("### Classes possíveis:")
    st.markdown("airplanes, bicycles, birds, boats, bottles, buses, cars, cats, chairs, cows, dining tables")
    st.markdown("dogs, horses, motorbikes,people, potted plants, sheep, sofas, trains, and tvs")
    st.markdown("### Defina o grau de confiança na classificação do modelo")

    img_file_buffer = st.file_uploader("Carregue uma imagem", type=["png", "jpg", "jpeg"])
    
    if img_file_buffer is not None:
       	image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    img_raw = image
  
    #confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    #img_file_buffer = st.file_uploader("Carregue uma imagem", type=["png", "jpg", "jpeg"])

    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    #if img_file_buffer is not None:
    #    image = np.array(Image.open(img_file_buffer))

    #else:
    #    demo_image = DEMO_IMAGE
    #    image = np.array(Image.open(demo_image))

    detections = process_image(image)
    image, labels = annotate_image(image, detections, confidence_threshold)
    st.image(image, caption=f"Processed image", use_column_width=True)
    st.markdown("### Classificação")
    #counter_dict = Counter(labels)
    #print(counter_dict.values())
    st.text(labels)


    #if st.button("Canny"):
    #    image = img_raw
    #    flag_canny=True
    #    edges = cv2.Canny(image,100,200)
    #    st.image(edges, caption=f"Edge image", use_column_width=True)

    #if st.button("Gray"):
    #    image = img_raw
    #    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #    st.image(gray, caption=f"Gray color", use_column_width=True)


    #if st.button("Detect eyes"):
    #    image = img_raw
    #    image = cv2.cvtColor(image,1)
    #    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #    st.text("Gray img for eyes")

    #    neighbors_threshold = st.slider("Number of neighbors", 0, 10, DEFAULT_NEIGHBORS_THRESHOLD, 1)

    #    eyes = eye_cascade.detectMultiScale(gray, 1.3, neighbors_threshold)
    #    print("Olhos",eyes)
        
    #    for (ex,ey,ew,eh) in eyes:
    #	        cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    #    st.image(image, caption=f"Eyes detected", use_column_width=True)


def detectar_olhos():
    img_file_buffer = st.file_uploader("Carregue uma imagem", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
       	image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    neighbors_threshold = st.slider("Neighbors threshold", 1, 15, DEFAULT_NEIGHBORS_THRESHOLD, 1)

    image = cv2.cvtColor(image,1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, 1.3, neighbors_threshold)
    print("Olhos",eyes)
        
    for (ex,ey,ew,eh) in eyes:
    	    cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    st.image(image, caption=f"Eyes detected", use_column_width=True)


def gray():
    img_file_buffer = st.file_uploader("Carregue uma imagem", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
       	image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
    img_raw = image
    st.image(img_raw, caption=f"Original", use_column_width=True)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    st.image(gray, caption=f"Gray color", use_column_width=True)

def canny():
    img_file_buffer = st.file_uploader("Carregue uma imagem", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
       	image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
    img_raw = image
    st.image(img_raw, caption=f"Original", use_column_width=True)
    edges = cv2.Canny(image,100,200)
    st.image(edges, caption=f"Edge image", use_column_width=True)



# Streamlit encourages well-structured code, like starting execution in a main() function.
st.title("OpenCV e MobileNet SSD")
st.markdown("### Modelo treinado com COCO dataset.")
st.markdown("### Ajustado com PASCAL VOC atingindo 72.7% map (mean average precision).")

st.markdown("========================================================================")
demo_image = DEMO_IMAGE
image = np.array(Image.open(demo_image))



if __name__ == "__main__":
    main()
