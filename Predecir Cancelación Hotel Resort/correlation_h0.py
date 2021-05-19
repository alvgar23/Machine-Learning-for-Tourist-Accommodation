# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 18:48:52 2021

@author: alvar
"""


"""
Calculamos la matriz de correlación para ver si existe correlación entre las
variables del dataset Hotel Resort para predecir cancelaciones.
"""

#Librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#cargamos datos
# df = pd.read_csv("hotel_bookings_fechas.csv")
df = pd.read_csv("dataset_pca.csv")


#Ahora debemos extraer todos los registros que no sean Resort Hotel y además
#extraer la columna hotel

#borramos las filas donde hotel=0 (resort hotel)
df.drop(df.index[39596:118898], axis=0, inplace=True)
df.drop('hotel',1, inplace=True) #borra columna del dataset



correlations = df.corr()


#matriz de correlación con mapa de calor
fig, ax = plt.subplots(figsize=(30,30))
# sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
#     square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
sns.heatmap(correlations, vmax=1, vmin=0, center=0.75, square=False)
plt.show();


#la mejor forma de que se vea la matriz es tight layout y ajustar un poco 
#en derecha y en izquierda
