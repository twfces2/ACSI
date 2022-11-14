import database as db
import streamlit as st
import numpy as np
import redneu as red
import pyautogui

all_files = np.array(db.enlistar())



st.set_page_config(page_title="DICOMS",page_icon=":tada:",layout="wide")

st.subheader("Diagnóstico automático de Derrame Pleural")

all_files = np.array(db.enlistar())

with st.container():
    st.write("---")
    option = st.selectbox(
        'Que archivo se leera',
        all_files)

    st.write('Seleccionaste:', option)

name=option 
iamge=db.dcm_img(name)
iamge2=np.array(iamge)
iamge3,prediction=red.predecir(iamge2)

with st.container():
    st.write("---")
    st.write("Patología Detectada: Efusion Pleura  Porcentaje de Certeza: "+str(prediction[0,5]))
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("Original")
        st.image(iamge,width=600)
    with right_column:
        st.write("Diagóstico")
        st.image(iamge3,width=600)

with st.container():
    file = st.file_uploader("Seleccione el archivo a subir")
    if file is not None:
        result = st.button("Subir archivo")
        if result:
            bytes_file = file.getvalue()
            db.insert_dcm("prueba.dcm",bytes_file)  
            pyautogui.hotkey("ctrl","F5")
            
with st.container():    
    cuadro=db.dcm_info(name)
    st.table(cuadro)
