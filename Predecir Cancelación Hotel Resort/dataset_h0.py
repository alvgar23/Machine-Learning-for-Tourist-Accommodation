# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:32:34 2021

@author: alvar
"""

"""
Paso 1: Borramos registros de hotel 0
Paso 2: Borramos variable hotel
Paso 3: Borramos variable reservation_status
Paso 4: Borramos variables: babies, previous_bookings_not_canceled, company,
        days_in_waiting_list, is_repeated_guest, children, 
        required_car_parking_spaces (Poca importancia de variable)
Paso 5: Borramos arrival_date_month, arrival_date_year, 
        arrival_date_day_of_month (por correlación)
Paso 6: Chequeamos variable importance
Paso 7: Creamos dataset final
"""



#importamos algunas librerías necesarias
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier



#Los datos que hallamos en el 1er análisis de datos (todo valores numéricos)
# df = pd.read_csv("hotel_bookings_fechas.csv")
df = pd.read_csv("dataset_pca.csv")

#borramos las filas donde hotel=0 (resort hotel)
df.drop(df.index[39596:118898], axis=0, inplace=True)



#Queremos crear 2 matrices: una contiene todos los datos (X) excepto los 
# de is_canceled que estarán en su propia matriz (y)
y_name = 'is_canceled'   #cabecera de la matriz y

X_name = df.columns.values.tolist() #creamos lista con cabecera de matriz X
X_name.remove(y_name)  #borramos cabecera y de X
X_name.remove('reservation_status') #no tiene sentido para predecir cancelación
X_name.remove('hotel') #borramos hotel (siempre es 0)
X_name.remove('babies')
X_name.remove('previous_bookings_not_canceled')
X_name.remove('company')
X_name.remove('days_in_waiting_list')
X_name.remove('is_repeated_guest')
X_name.remove('arrival_date_month')
X_name.remove('arrival_date_year')
X_name.remove('arrival_date_day_of_month')


y = df[y_name].tolist() #creamos una lista con los valores de y
y = np.array(y)   #convertimos la lista a array tipo numpy



#Antes de trabajar con los datos, debemos convertir datos a tipo numpy
X = df.drop(y_name,1, inplace=True) # borra y del data set X
X = df.drop('reservation_status',1, inplace=True) #borra columna del dataset
X = df.drop('hotel',1, inplace=True) #borra columna del dataset
X = df.drop('babies',1, inplace=True)
X = df.drop('previous_bookings_not_canceled',1, inplace=True)
X = df.drop('company',1, inplace=True)
X = df.drop('days_in_waiting_list',1, inplace=True)
X = df.drop('is_repeated_guest',1, inplace=True)
X = df.drop('arrival_date_month',1, inplace=True)
X = df.drop('arrival_date_year',1, inplace=True)
X = df.drop('arrival_date_day_of_month',1, inplace=True)
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



#Volvemos a añadir is_canceled al dataset final
df['is_canceled'] = y


#dataset final
print(df.info())

# Creamos un nuevo archivo csv con el dataset final
df.to_csv("dataset_final_h0.csv", index=False)

