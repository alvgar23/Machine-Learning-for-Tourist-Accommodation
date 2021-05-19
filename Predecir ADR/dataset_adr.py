# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:32:37 2021

@author: alvar
"""

"""
Paso 1: Borramos variables: babies, previous_bookings_not_canceled, company,
        previous_cancellations, days_in_waiting_list, is_canceled, 
        reservation_status, (Poca importancia de variable) 
Paso 2: Borramos reservation_status_date, arrival_date_year y 
        arrival_date_day_of_month (por correlación)
Paso 3: Chequeamos variable importance
Paso 4: Creamos dataset final
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
X_name.remove('babies')
X_name.remove('previous_bookings_not_canceled')
X_name.remove('company')
X_name.remove('days_in_waiting_list')
X_name.remove('is_canceled')
X_name.remove('previous_cancellations')
X_name.remove('reservation_status')
X_name.remove('required_car_parking_spaces')
X_name.remove('total_of_special_requests')
X_name.remove('arrival_date_year')
X_name.remove('reservation_status_date')
X_name.remove('arrival_date_day_of_month')



y = df[y_name].tolist() #creamos una lista con los valores de y
y = np.array(y)   #convertimos la lista a array tipo numpy



#Antes de trabajar con los datos, debemos convertir datos a tipo numpy
X = df.drop(y_name,1, inplace=True) # borra y del data set X
X = df.drop('babies',1, inplace=True)
X = df.drop('previous_bookings_not_canceled',1, inplace=True)
X = df.drop('company',1, inplace=True)
X = df.drop('previous_cancellations',1, inplace=True) #borra columna del dataset
X = df.drop('days_in_waiting_list',1, inplace=True)
X = df.drop('is_canceled',1, inplace=True)
X = df.drop('reservation_status',1, inplace=True) #borra columna del dataset
X = df.drop('required_car_parking_spaces',1, inplace=True)
X = df.drop('total_of_special_requests',1, inplace=True)
X = df.drop('arrival_date_year',1, inplace=True)
X = df.drop('reservation_status_date',1, inplace=True)
X = df.drop('arrival_date_day_of_month',1, inplace=True)
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



#Volvemos a añadir adr al dataset final
df['adr'] = y



#dataset final
print(df.info())

# Creamos un nuevo archivo csv con el dataset final
df.to_csv("dataset_final_adr.csv", index=False)