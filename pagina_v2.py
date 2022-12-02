import pickle
from pathlib import Path

import pandas as pd  
import plotly.express as px  
import streamlit as st 
import streamlit_authenticator as stauth

import database as db
import numpy as np
import redneu as red



# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
#st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

st.set_page_config(page_title="DICOMS",page_icon=":tada:",layout="wide")

st.subheader("Diagnóstico automático de Derrame Pleural")

# --- USER AUTHENTICATION ---
names = ["Mhax Tello", "Brad Guzman"]
usernames = ["MTello", "BGuzman"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    
    # ---- SIDEBAR ----
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome Dr. {name}")


    # ---- MAINPAGE ----
    st.title(":bar_chart: Casos de derrame pleural")
    st.markdown("##")
    
    
    all_files = np.array(db.enlistar())
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
                nombre=file.name
                bytes_file = file.getvalue()
                db.insert_dcm(nombre,bytes_file)  
                st.experimental_rerun()
            
    with st.container():    
        cuadro=db.dcm_info(name)
        st.table(cuadro)


