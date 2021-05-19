# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 20:40:38 2021

@author: alvar
"""

"""
Queremos hallar las variables más importantes a la hora de identificar el 
si una reserva está cancelada o no.
"""


#importamos algunas librerías necesarias
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


#Los datos que hallamos en el 1er análisis de datos (todo valores numéricos)
# df = pd.read_csv("hotel_bookings_fechas.csv")
df = pd.read_csv("dataset_pca.csv")


#Queremos crear 2 matrices: una contiene todos los datos (X) excepto los 
# de is_canceled que estarán en su propia matriz (y)
y_name = 'is_canceled'   #cabecera de la matriz y

X_name = df.columns.values.tolist() #creamos lista con cabecera de matriz X
X_name.remove(y_name)  #borramos cabecera y de X
X_name.remove('reservation_status') #no tiene sentido para predecir cancelación

y = df[y_name].tolist() #creamos una lista con los valores de y
y = np.array(y)   #convertimos la lista a array tipo numpy



#Antes de trabajar con los datos, debemos convertir datos a tipo numpy
X = df.drop(y_name,1, inplace=True) # borra y del data set X
X = df.drop('reservation_status',1, inplace=True) #borra columna del dataset
X = df.to_numpy() # convierte en matriz numpy



#Entrenamos el modelo de variable importance
model = ExtraTreesClassifier()
model.fit(X, y)



#y para visualizar resultados
var_imp = (pd.DataFrame({
 'feature': X_name, 
 'v_importance':model.feature_importances_.tolist()
 }))

print(var_imp.sort_values(by = 'v_importance', ascending=False))