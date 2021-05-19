# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:47:56 2021

@author: alvar
"""

"""
Queremos hallar las variables más importantes a la hora de predecir el adr.
En este caso no es un problema de clasificación, sino de regresión, 
por lo tanto no usamos ExtraTreesClassifier sino ExtraTreesRegressor
"""


#importamos algunas librerías necesarias
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


#Los datos que hallamos en el 1er análisis de datos (todo valores numéricos)
# df = pd.read_csv("hotel_bookings_fechas.csv")
df = pd.read_csv("dataset_pca.csv")


#Queremos crear 2 matrices: una contiene todos los datos (X) excepto los 
# de adr que estarán en su propia matriz (y)
y_name = 'adr'   #cabecera de la matriz y

X_name = df.columns.values.tolist() #creamos lista con cabecera de matriz X
X_name.remove(y_name)  #borramos cabecera y de X

y = df[y_name].tolist() #creamos una lista con los valores de y
y = np.array(y)   #convertimos la lista a array tipo numpy



#Antes de trabajar con los datos, debemos convertir datos a tipo numpy
X = df.drop(y_name,1, inplace=True) # borra y del data set X
X = df.to_numpy() # convierte en matriz numpy



#Entrenamos el modelo de variable importance
model = ExtraTreesRegressor()
model.fit(X, y)



#y para visualizar resultados
var_imp = (pd.DataFrame({
 'feature': X_name, 
 'v_importance':model.feature_importances_.tolist()
 }))

print(var_imp.sort_values(by = 'v_importance', ascending=False))