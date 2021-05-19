# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:57:03 2021

@author: alvar
"""

"""
______________________________________________________________________________
En este documento vamos a crear una función que nos permita consultar
algunos de los datos existentes en cada uno de los conjuntos de datos que
tengamos.
______________________________________________________________________________
También vamos a anotar indices interesantes de nuestros conjuntos de datos.
______________________________________________________________________________
"""


import os
import pandas as pd
from time import sleep


def mostrar_directorios():

    #Guarda los directorios en una lista
    directorios = os.listdir()
    
    print("Estos son los conjuntos de datos disponibles: \n")
    
    #mostramos solo los .csv 
    for direc in directorios:
        if "csv" in direc:
            print(direc)
    
    return directorios


def consultar_dato(df):
    
    resp = "?"
    
    while resp != "f" and resp != "c" and resp != "a":
        resp = input("¿Quiere consultar una fila, una columna o ambas? (f/c/a)\n")
    
    if resp == "f":
        print(0, "-", len(df), "filas")
        fil = int(input("Introduzca la fila: "))
        print(df.iloc[fil])
    if resp == "c":
        print(df.info())
        col = input("Introduzca la columna sin comillas: ")
        print(df[col])
    if resp == "a":    
        print(0, "-", len(df), "filas")
        fil = int(input("Introduzca la fila: "))
        print(df.info())
        col = input("Introduzca la columna sin comillas: ")
        print(df.loc[fil, [col]])

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
    df = pd.read_csv(archivo)
    sleep(0.5)
    print(archivo, "cargado con éxito")


consultar_dato(df)


"""
Datos interesantes:
    0-40059: Resort Hotel
    40060-119390: City Hotel
    40600-40667-40679-41160: Children = NaN
"""