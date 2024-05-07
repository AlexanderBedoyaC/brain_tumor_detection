import numpy as np
import pandas as pd
import tensorflow as tf

import os
from os import listdir ### para hacer lista de archivos en una ruta
from tqdm import tqdm  ### para crear contador en un for para ver evolución
import cv2 ### para leer imagenes jpg
import openpyxl

import sys
sys.path.append('utils/')

if __name__=="__main__":
    
    path = os.getcwd() + '\despliegue\\'
    
        
    print(listdir(path))

    x = []   #### una lista con el array que representa cada imágen
    names = []
    
    list_labels = [path + f for f in listdir(path)] ### crea una lista de los archivos en la ruta (no / yes)

    for imagePath in tqdm(list_labels): ### recorre cada carpeta de la ruta ingresada
        if imagePath[-1] =='g': ### verificar que se imágen extensión jpg o jpeg
            img = cv2.imread(imagePath) ### cargar archivo
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) ### invierte el orden de los colores en el array para usar el más estándar RGB
            img = cv2.resize(img ,(100, 100)) ### cambia resolución de imágnenes
            x.append(img) ###adiciona imágen al array final
            names.append(imagePath)

    x = np.array(x) # imagenes a predecir

    x = x.astype('float') # convertir para escalar
    x /= 255 # escalar datos


    files2 = [name.rsplit('.', 1)[0] for name in names] ### eliminar extension a nombre de archivo
    files2 = [name.rsplit('\\')[-1] for name in files2] ### Tomar solo la identificación del paciente

    modelo = tf.keras.models.load_model(os.getcwd() + '/salidas/best_model.h5') ### cargar modelo
    prob = modelo.predict(x)

    clase = ['2. Tumor' if prob >= 0.98 else '3. No tumor' if prob <= 0.02 else "1. Revisión manual" for prob in prob]
    
    mensaje_no_tumor = '''¡Buenas noticias! Su resonancia magnética no muestra presencia de ningún tumor.
    Siéntase tranquilo/a, su salud está en buen estado. No obstante, continúe con sus chequeos regulares para mantenerse saludable.'''
    
    mensaje_revision_manual = '''Su resonancia magnética ha arrojado resultados que requieren una revisión manual por 
    parte de un especialista. Esto puede deberse a varios factores y no necesariamente indica la presencia de un tumor. 
    Le recomendamos que se ponga en contacto con su médico lo antes posible para obtener más información y orientación.'''

    mensaje_si_tumor = '''Nos entristece informarle que hemos detectado ciertas anomalías en su resonancia magnética que sugieren la
    presencia de un tumor. Sabemos que esta noticia puede ser abrumadora, pero es crucial que tome medidas rápidas. 
    Le recomendamos que se comunique de inmediato con su médico para comenzar la evaluación y discutir las opciones 
    de tratamiento. Estamos aquí para brindarle apoyo en cada paso del proceso. Su bienestar es nuestra prioridad'''
    
    mensaje = [mensaje_si_tumor if prob >= 0.98 else mensaje_no_tumor if prob <= 0.02 else mensaje_revision_manual for prob in prob]
    
    probabilidades = [prob for prob in prob.reshape(1, -1)[0]]
    
    res_dict = {
        "Paciente": files2,
        "Clase": clase,
        "Mensaje" : mensaje,
        "Probabilidad" : probabilidades
    }
    
    resultados = pd.DataFrame(res_dict)
    
    resultados = resultados.sort_values(by = ['Clase', 'Probabilidad'], ascending = [True, False])
    
    resultados['Probabilidad'] = resultados['Probabilidad'].apply(lambda x: str(round(x*100, 2)) + '%')

    resultados.to_excel(os.getcwd() + '/salidas/resultados.xlsx', index = False)