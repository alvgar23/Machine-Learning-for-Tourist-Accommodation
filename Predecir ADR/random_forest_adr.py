# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:04:17 2021

@author: alvar
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    
    #Entrenamos el modelo
    regression = RandomForestRegressor(random_state=0)
    regression.fit(X_train, y_train)
    
    #Error del entrenamiento del modelo
    pred_train = regression.predict(X_train)
    MAE = mean_absolute_error(pred_train, y_train)
    MSE = mean_squared_error(pred_train, y_train)
    print("--------------TRAIN----------------")
    print("Mean Absolute Error:", MAE) 
    print("Mean Squared Error:", MSE)
    
    
    
    #Predecimos en nuestro 20% de test
    predictions = regression.predict(X_test)
    
    
    
    #performance against test data (error)
    print("-------TEST--------")
    mean_absolute_error_test = mean_absolute_error(y_test, predictions)
    print("Mean Absolute Error:", mean_absolute_error_test)
    
    mean_squared_error(y_test, predictions)
    mean_squared_error_test = math.sqrt(mean_squared_error(y_test, predictions))
    print("Mean Squared Error:", mean_squared_error_test)
    
    
    #algunas predicciones
    print("---------TEST---------")
    print("Predicciones:",predictions)
    print("Real:", y_test)
    
    generar_excel(predictions)





###DESDE AQUÍ EMPIEZO A USAR GRID SEARCH PARA OPTIMIZAR HIPERPARÁMETROS




def buscar_parametros():

    #Defino parámetros aleatorios para Random Forest. RandomizeSearchCV los 
    #optimizará luego
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    
    
    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
    
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    grid = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
                n_iter = 80, cv = 2, verbose=300, random_state=0)



"""
Estos han sido los mejores parámetros de la randomización de los parámetros.
Se han seleccionado de manera aleatoria los siguientes parámetros:
    {'bootstrap': False,
      'max_depth': 30,
      'max_features': sqrt,
      'min_samples_leaf': 5,
      'min_samples_split': 5,
      'n_estimators': 100}
Trataremos de incrementar la precisión aún más utilizando valores aproximados
y haciendo un gridsearch con ellos.
"""



def optimizado():

    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [False],
        'max_depth': [None],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1],
        'min_samples_split': [2],
        'n_estimators': [500]
                  }
    
    
    
    
    #Indico modelo, parámetros, refit=True para optimizar parámetros
    grid = GridSearchCV(RandomForestRegressor(random_state=0), 
                        param_grid, refit=True, cv=2, verbose=50)
    
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
    mean_absolute_error_test = mean_absolute_error(y_test, grid_predictions)
    print("Mean Absolute Error:", mean_absolute_error_test)
    
    mean_squared_error(y_test, grid_predictions)
    mean_squared_error_test = math.sqrt(mean_squared_error(y_test, grid_predictions))
    print("Mean Squared Error:", mean_squared_error_test)
    
    generar_excel(grid_predictions)
    
    
    
    
    
#Generamos EXCEL con los resultados
def generar_excel(predictions):
    df_y_test = pd.DataFrame(y_test)
    df_y_test.to_excel('y_test_adr_rf.xlsx', index=False, header=True)
    
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_excel('predictions_adr_rf.xlsx', index=False, header=True)





menu ="""
1. Sin Optimizar.
2. Buscar parámetros para optimizar.
3. Optimizado. 
"""

print(menu)
respuesta = int(input("Elija una opción: "))

if respuesta == 1:
    sin_optimizar()
if respuesta == 2:
    buscar_parametros()
if respuesta == 3:
    optimizado()