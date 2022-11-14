from pydicom import dcmread
import matplotlib.pyplot as plt
from deta import Deta
from io import BytesIO
from PIL import Image
import numpy as np
import os
import pandas as pd


DETA_KEY = "e0ff1r3f_xoNt6bMZJceHhxN37sKmcQhzwtcK4aQt"
# initialize with a project key
deta = Deta(DETA_KEY)

# create and use as many Drives as you want!
db = deta.Drive("PACS")


def enlistar():
    result = db.list()
    all_files = result.get("names")
    return all_files

def insert_dcm(name,file):
    return db.put(name, BytesIO(file)) 

def getdcm(name):
    archiv= db.get(name).read()
    ds = dcmread(BytesIO(archiv))
    return ds

def dcm_img(nombre):
    ds = getdcm(nombre)
    im = ds.pixel_array.astype(float)
    rescaled_image = (np.maximum(im,0)/im.max())*255
    final_image = np.uint8(rescaled_image)
    final_image = Image.fromarray(final_image)
    return final_image

def dcm_info(nombre):
    ds=getdcm(nombre)
    llaves=list(ds.keys())
    cuadro=[]
    for i in range(0,len(llaves)-1):
        fila=[]
        fila.append(ds[llaves[i]].name)
        fila.append(ds[llaves[i]].value)
        cuadro.append(fila)
    metadata = pd.DataFrame(cuadro,columns=['Categoria','Valor'])
    return metadata
    
