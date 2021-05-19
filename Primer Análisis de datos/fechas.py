# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:45:22 2021

@author: alvar
"""


"""
______________________________________________________________________________
En este paso vamos a manejar las fechas y establecemos las bases para tener
series temporales.
______________________________________________________________________________
Lo importante es crear números enteros que sigan el orden año, mes, dia en un
formato del tipo 20170130. Así puede ordenarse sin problemas por fecha.
______________________________________________________________________________
Después vamos a crear 2 nuevas variables:
    full_date: Un entero que combina arrival_date_year y week_number
    arrival_date: Fecha que combina arrival_date_year, month y day
______________________________________________________________________________
De esta manera tendremos dos variables que nos indican claramente la fecha de 
llegada y la semana del año, para poder observar la estacionalidad.
______________________________________________________________________________
"""


import pandas as pd


#leemos los datos transformados
df_conv = pd.read_csv("hotel_bookings_enteros.csv")


#convertimos dato string en dato fecha
# df_conv["reservation_status_date"] = (pd.to_datetime(
#     df_conv["reservation_status_date"], format="%Y-%m-%d"))



#Primeros pasos para crear full_date
#Todos los elementos de week y year serán string
df_conv["arrival_date_week_number"] = df_conv["arrival_date_week_number"].map(str)
week = df_conv["arrival_date_week_number"]
df_conv["arrival_date_year"] = df_conv["arrival_date_year"].map(str)
year = df_conv["arrival_date_year"]


#Primeros pasos para crear arrival_date
#Todos los elementos de day y month serán string
df_conv["arrival_date_day_of_month"] = df_conv["arrival_date_day_of_month"].map(str)
day = df_conv["arrival_date_day_of_month"]
df_conv["arrival_date_month"] = df_conv["arrival_date_month"].map(str)
month = df_conv["arrival_date_month"]



#ejemplo: transforma '1' en '01', pero '10' lo deja igual.
df_conv["arrival_date_week_number"].replace("1","01",inplace=True)
df_conv["arrival_date_week_number"].replace("2","02",inplace=True)
df_conv["arrival_date_week_number"].replace("3","03",inplace=True)
df_conv["arrival_date_week_number"].replace("4","04",inplace=True)
df_conv["arrival_date_week_number"].replace("5","05",inplace=True)
df_conv["arrival_date_week_number"].replace("6","06",inplace=True)
df_conv["arrival_date_week_number"].replace("7","07",inplace=True)
df_conv["arrival_date_week_number"].replace("8","08",inplace=True)
df_conv["arrival_date_week_number"].replace("9","09",inplace=True)
df_conv["arrival_date_month"].replace("1","01",inplace=True)
df_conv["arrival_date_month"].replace("2","02",inplace=True)
df_conv["arrival_date_month"].replace("3","03",inplace=True)
df_conv["arrival_date_month"].replace("4","04",inplace=True)
df_conv["arrival_date_month"].replace("5","05",inplace=True)
df_conv["arrival_date_month"].replace("6","06",inplace=True)
df_conv["arrival_date_month"].replace("7","07",inplace=True)
df_conv["arrival_date_month"].replace("8","08",inplace=True)
df_conv["arrival_date_month"].replace("9","09",inplace=True)
df_conv["arrival_date_day_of_month"].replace("1","01",inplace=True)
df_conv["arrival_date_day_of_month"].replace("2","02",inplace=True)
df_conv["arrival_date_day_of_month"].replace("3","03",inplace=True)
df_conv["arrival_date_day_of_month"].replace("4","04",inplace=True)
df_conv["arrival_date_day_of_month"].replace("5","05",inplace=True)
df_conv["arrival_date_day_of_month"].replace("6","06",inplace=True)
df_conv["arrival_date_day_of_month"].replace("7","07",inplace=True)
df_conv["arrival_date_day_of_month"].replace("8","08",inplace=True)
df_conv["arrival_date_day_of_month"].replace("9","09",inplace=True)




#Función para crear full_date
def fulldate(fila):
    week = fila["arrival_date_week_number"] #son datos str 
    year = fila["arrival_date_year"] #son datos str
    resultado = (year + week) #ejemplo: resultado = '201507' =>201507  
    return resultado



#Función para crear arrival_date
def arrivaldate(fila):
    year = fila["arrival_date_year"]   #son datos str 
    month = fila["arrival_date_month"]   #son datos str 
    day = fila["arrival_date_day_of_month"]   #son datos str 
    
    resultado = year + month + day # es un str
    return resultado


#Función para convertir reservation_status_date
def statusdate(fila):
    status = fila["reservation_status_date"] #datos str tipo "year-month-day"
    status = status.split("-")  #borramos los "-" y separamos cada elemento
    status = "".join(status)   #lo juntamos todo en un nuevo str "yearmonthday"
    return status


#Creamos full_date y arrival_date
df_conv["full_date"] = df_conv.apply(fulldate, axis=1)
df_conv["arrival_date"] = df_conv.apply(arrivaldate, axis=1)
df_conv["reservation_status_date"] = df_conv.apply(statusdate, axis=1)



#volvemos a convertir a enteros todos los datos que habíamos transformado
df_conv["arrival_date_week_number"] = week.map(int)
df_conv["arrival_date_year"] = year.map(int)
df_conv["arrival_date_day_of_month"] = day.map(int)
df_conv["arrival_date_month"] = month.map(int)
df_conv["full_date"] = df_conv["full_date"].map(int)
df_conv["arrival_date"] = df_conv["arrival_date"].map(int)
df_conv["reservation_status_date"] = df_conv["reservation_status_date"].map(int)


# Creamos un nuevo archivo csv con estos nuevos datos transformados
df_conv.to_csv("hotel_bookings_fechas.csv", index=False)



print(df_conv.info())





# Parece que al cargarlo de nuevo los datos datetime aparecen como object
# df_test = pd.read_csv("hotel_bookings_fechas.csv")
# print(df_test.info())
#Para solucionarlo podemos incluir estas lineas cuando necesitemos una serie
#temporal
# df["reservation_status_date"]= (pd.to_datetime(
#   df["reservation_status_date"], format="%Y-%m-%d"))
# df["arrival_date"]=pd.to_datetime(df["arrival_date"], format="%Y-%m-%d")