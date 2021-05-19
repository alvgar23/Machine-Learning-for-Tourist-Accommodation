# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:12:55 2021

@author: alvar
"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



#Leemos los datos (20)
df = pd.read_csv("dataset_final_adr.csv")



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



#asignamos el 20% a test (el 80% restante se queda en train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                  random_state=0)






###A PARTIR DE AQUÍ EMPIEZA EL CÓDIGO SIN OPTIMIZAR





def sin_optimizar():

    #Entrenamos el modelo para 3 diferentes valores de epsilon
    svr_reg_0 = LinearSVR(epsilon=0)
    svr_reg_05 = LinearSVR(epsilon=0.5)
    svr_reg_15 = LinearSVR(epsilon=1.5)
    
    svr_reg_0.fit(X_train, y_train)
    svr_reg_05.fit(X_train, y_train)
    svr_reg_15.fit(X_train, y_train)
    
    
    
    #Error del entrenamiento del modelo
    pred_train0 = svr_reg_0.predict(X_train)
    pred_train05 = svr_reg_05.predict(X_train)
    pred_train15 = svr_reg_15.predict(X_train)
    
    MAE0 = mean_absolute_error(pred_train0, y_train)
    MSE0 = mean_squared_error(pred_train0, y_train)
    MAE05 = mean_absolute_error(pred_train05, y_train)
    MSE05 = mean_squared_error(pred_train05, y_train)
    MAE15 = mean_absolute_error(pred_train15, y_train)
    MSE15 = mean_squared_error(pred_train15, y_train)
    print("--------------TRAIN----------------")
    print("Mean Absolute Error 0:", MAE0) 
    print("Mean Squared Error 0:", math.sqrt(MSE0))
    print("Mean Absolute Error 05:", MAE05) 
    print("Mean Squared Error 05:", math.sqrt(MSE05))
    print("Mean Absolute Error 15:", MAE15) 
    print("Mean Squared Error 15:", math.sqrt(MSE15))
    
    
    # LinearSVR(C=1.0, dual=True, epsilon=1.5, fit_intercept=True,
    #           intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
    #           random_state=None, tol=0.0001, verbose=0)
    
    
    
    #predicciones con test data
    predictions0 = svr_reg_0.predict(X_test)
    predictions05 = svr_reg_05.predict(X_test)
    predictions15 = svr_reg_15.predict(X_test)
    
    
    
    #performance against test data (error)
    #Elegimos el epsilon que nos haya dado un error más bajo
    #Epsilon 0.5 y epsilon 15 están muy igualados, asi que probamos ambos
    print("-------TEST--------")
    mean_absolute_error_p0 = mean_absolute_error(y_test, predictions0)
    mean_absolute_error_p05 = mean_absolute_error(y_test, predictions05)
    mean_absolute_error_p15 = mean_absolute_error(y_test, predictions15)
    print("Mean Absolute Error p0:", mean_absolute_error_p0)
    print("Mean Absolute Error p05:", mean_absolute_error_p05)
    print("Mean Absolute Error p15:", mean_absolute_error_p15)
    
    
    mean_squared_error(y_test, predictions0)
    mean_squared_error_p0 = math.sqrt(mean_squared_error(y_test, predictions0))
    print("Mean Squared Error p0:", mean_squared_error_p0)
    
    mean_squared_error(y_test, predictions05)
    mean_squared_error_p05 = math.sqrt(mean_squared_error(y_test, predictions05))
    print("Mean Squared Error p05:", mean_squared_error_p05)
    
    mean_squared_error(y_test, predictions15)
    mean_squared_error_p15 = math.sqrt(mean_squared_error(y_test, predictions15))
    print("Mean Squared Error p15:", mean_squared_error_p15)
    
    
    #algunas predicciones (con epsilon=1,5 porque dio el error más bajo)
    print("---------TEST---------")
    print("Predicciones-> \n p0:",predictions0, "\np05:", predictions05, "\np15:", predictions15)
    print("Real:", y_test)


    #Modificamos por 05 y 15 para almacenar todas las predicciones
    #en distintos documentos. Cuidado con el nombre para no sobreescribir
    generar_excel(predictions0)







###DESDE AQUÍ EMPIEZO A USAR GRID SEARCH PARA OPTIMIZAR HIPERPARÁMETROS







def optimizado():

    #Defino parámetros para SVM. GridSearch los optimizará luego
    #Originalmente: C: 0.1, 1, 10, 100, 1000;   gamma: 1, 0.1, 0.01, 0.001, 0.0001
    param_grid = {'C': [0.1], 
                  'epsilon': [1],
                  } 
    
    #Indico modelo, parámetros, refit=True para optimizar parámetros
    grid = GridSearchCV(LinearSVR(), 
                        param_grid, refit=True, verbose=100, cv=5)
    
    import time  # Just to compare fit times
    start = time.time()
    #Entreno el modelo para la búsqueda de grid
    grid.fit(X_train, y_train)
    end = time.time()
    print("GridSearch Fit Time:", end - start)
    
    
    
    # print best parameter after tuning
    print(grid.best_params_)
      
    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    
    #Hacemos predicciones con data test
    grid_predictions = grid.predict(X_test)
    
    #Resultados obtenidos
    mae = mean_absolute_error(y_test, grid_predictions)
    mean_squared_error(y_test, grid_predictions)
    mean_squared_error_sqrt = math.sqrt(mean_squared_error(y_test, grid_predictions))
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mean_squared_error_sqrt)

    generar_excel(grid_predictions)






#Generamos EXCEL con los resultados
def generar_excel(predictions):
    df_y_test = pd.DataFrame(y_test)
    df_y_test.to_excel('y_test_adr_svm.xlsx', index=False, header=True)
    
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_excel('predictions_adr_svm.xlsx', index=False, header=True)




menu ="""
1. Sin Optimizar.
2. Optimizado. 
"""

print(menu)
respuesta = int(input("Elija una opción: "))

if respuesta == 1:
    sin_optimizar()
if respuesta == 2:
    optimizado()