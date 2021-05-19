# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 00:51:15 2021

@author: alvar
"""


"""
______________________________________________________________________________
En este paso vamos a limpiar y filtrar algunos datos que hemos
observado que tienen valores nulos.
______________________________________________________________________________
La columna "Company" y "Agent" tienen valores nulos que han de ser sustituidos
por el valor 0 (sin empresa, sin agente).
______________________________________________________________________________
En el caso de la columna "children", nos conviene eliminar las
filas con datos nulos, ya que NA no asegura que children=0.
______________________________________________________________________________
También hay valores nulos en la nacionalidad que también hay que borrar.
______________________________________________________________________________
"""


import pandas as pd


df = pd.read_csv("hotel_bookings.csv")


tabla = df.info()
print(tabla) #muestra datos nulos respecto a datos totales d cada columna


#Cambiamos valores nulos de "agent" y "company" por el valor 0.
df.fillna({"agent":0, "company":0}, inplace=True)


#Borramos las 4 filas con children="NA" y las NULL de nationality.
#Deberían quedar 118898 datos
df.dropna(how='any', inplace=True)


#Convertimos children, agent, company de tipo float a tipo int
df["children"] = df["children"].astype('int64')
df["agent"] = df["agent"].astype('int64')
df["company"] = df["company"].astype('int64')

#Creamos un nuevo archivo csv con estos nuevos datos filtrados
#index=False evita crear automáticamente una nueva columna sin nombre con id
df.to_csv("hotel_bookings_filtrado.csv", index=False)


#leemos el nuevo archivo para chequear
df_filtrado = pd.read_csv("hotel_bookings_filtrado.csv")


#Solo si no hemos puesto index=False antes, para borrar la nueva columna
# df_filtrado.drop(['Unnamed: 0'], axis=1, inplace=True)


print(df_filtrado.info())
