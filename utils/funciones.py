from kaggle.api.kaggle_api_extended import KaggleApi

from os import listdir ### para hacer lista de archivos en una ruta
from tqdm import tqdm  ### para crear contador en un for para ver evolución
from os.path import join ### para unir ruta con archivo 
import cv2 ### para leer imagenes jpg

import numpy as np

def download_data(dataset):

    # Crea una instancia de la API de Kaggle
    api = KaggleApi()

    # Autentica con la API utilizando tu archivo de configuración kaggle.json
    api.authenticate()

    # Descarga el dataset especificando la ruta de destino y si deseas descomprimirlo
    api.dataset_download_files(dataset, path='data/', unzip=True)

def import_data(path, width = 100):
    
    rawImgs = []   #### una lista con el array que representa cada imágen
    labels = [] ### el label de cada imágen
    names = []
    
    list_labels = [path + f for f in listdir(path)] ### crea una lista de los archivos en la ruta (no / yes)

    for imagePath in ( list_labels): ### recorre cada carpeta de la ruta ingresada
        
        files_list=listdir(imagePath) ### crea una lista con todos los archivos
        for item in tqdm(files_list): ### le pone contador a la lista: tqdm
            file = join(imagePath, item) ## crea ruta del archivo
            if file[-1] =='g': ### verificar que se imágen extensión jpg o jpeg
                img = cv2.imread(file) ### cargar archivo
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) ### invierte el orden de los colores en el array para usar el más estándar RGB
                img = cv2.resize(img ,(width,width)) ### cambia resolución de imágnenes
                rawImgs.append(img) ###adiciona imágen al array final
                names.append(file)
                l = imagePath.split('/')[3] ### identificar en qué carpeta está
                if l == 'no':  ### verificar en qué carpeta está para asignar el label
                    labels.append([0])
                elif l == 'yes':
                    labels.append([1])
    return np.array(rawImgs), np.array(labels), names