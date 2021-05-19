# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:52:54 2021

@author: alvar
"""


"""
______________________________________________________________________________
Primero, vamos a cargar una lista con todos los valores existentes
sin repetición que hay en cada una de las columnas.
______________________________________________________________________________
Los ordenamos tal que el nombre de la columna aparece en primer lugar y los
valores nulos o indefinidos en el segundo lugar.
______________________________________________________________________________
La lista value_list contendrá estos valores.
Por pantalla se imprimirán los índices que le corresponden a cada variable.
Por ejemplo, value_list[0] mostrará los valores de la variable Hotel. 
______________________________________________________________________________
"""

import csv
import os
from time import sleep
import pandas as pd


value_list = []
for i in range(34):  #Hay 32 variables, en el de fechas 34
    value_list.append([])


def mostrar_directorios():

    #Guarda los directorios en una lista
    directorios = os.listdir()
    
    print("Estos son los conjuntos de datos disponibles: \n")
    
    #mostramos solo los .csv 
    for direc in directorios:
        if "csv" in direc:
            print(direc)
    
    return directorios



directorios = mostrar_directorios()
archivo = input("Introduzca el nombre del archivo que quiera abrir: \n")


while archivo not in directorios:
    print("Ese archivo no existe.\n")
    print("______________________________________________________________")
    mostrar_directorios()
    archivo = input("Introduzca el nombre del archivo que quiera abrir: \n")


if archivo in directorios:
    print("Cargando archivo...")
    sleep(0.5)
    with open(archivo, newline='') as File:  
        reader = csv.reader(File)
        sleep(0.5)
        print(archivo, "cargado con éxito")
        
        for row in reader:
            for col in range(len(value_list)):   
                if row[col] not in value_list[col]:
                    value_list[col].append(row[col])

            
        ordenar = input("¿Quieres ordenar los datos? (s/n)")
        if ordenar.lower() == "s":
            for col in range(len(value_list)): 
                value_list[col].sort(reverse=True)



df = pd.read_csv(archivo)
print(df.info())

        
# value_list = []
# for i in range(32):  #Hay 32 variables
#     value_list.append([])

# with open('hotel_bookings.csv', newline='') as File:  
#     reader = csv.reader(File)
    
#     #imprimimos los valores de cada variable
#     for row in reader:
#         for col in range(len(value_list)):   
#             if row[col] not in value_list[col]:
#                 value_list[col].append(row[col])
#             value_list[col].sort(reverse=True)



# La tabla de value_list en variable explorer irá a anexos


