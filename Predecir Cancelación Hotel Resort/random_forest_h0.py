# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:02:28 2021

@author: alvar
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


#Leemos los datos (22)
df = pd.read_csv("dataset_final_h0.csv")



#Queremos crear 2 matrices: una contiene todos los datos (X) excepto los 
# de is_canceled que estarán en su propia matriz (y)
y_name = 'is_canceled'   #cabecera de la matriz y

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

    #Se puede agregar un peso de clase "equilibrado" a la configuración de SVM, 
    #lo que agrega una penalización mayor a las clasificaciones incorrectas en la 
    #clase menor (en este caso, la clase de cancelación).
    clf = RandomForestClassifier(class_weight = 'balanced', random_state=0)
    clf.fit(X_train, y_train)
    
    #Precisión del entrenamiento del modelo
    score = clf.score(X_train, y_train)
    print("--------------TRAIN----------------")
    print("Precisión entrenamiento: ", score)
    
    
    #Ahora usamos Cross Validation para comparar con train
    #Si los resultados son parecidos, el modelo funciona correctamente
    kf = KFold(n_splits=5)
    score_cv = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy")
    print("--------------CROSS VALIDATION-----------")
    print("Precisión cv: ", score_cv)
    print("Precisión cv (valor medio): ", score_cv.mean())
    
    
    #Predecimos en nuestro 20% de test
    predictions = clf.predict(X_test)
    
    
    
    #Imprimimos los valores de precision, recall y f1 score. Probamos con test
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test ,predictions))
    
    
    #Precisión modelo y algunas predicciones
    print("---------TEST---------")
    accuracy = accuracy_score(y_test, predictions)
    print("Predicciones:",predictions)
    print("Real:", y_test)
    print("Precisión", accuracy)

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
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    grid = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
                n_iter = 100, cv = 3, verbose=10, random_state=0)






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
        'max_depth': [40],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1],
        'min_samples_split': [5],
        'n_estimators': [800]
                  }
    
    
    
    
    #Indico modelo, parámetros, refit=True para optimizar parámetros
    grid = GridSearchCV(RandomForestClassifier(class_weight = 'balanced',
                    random_state=0), param_grid, refit=True, cv=2, verbose=72)
    
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
    print(confusion_matrix(y_test, grid_predictions))
    print(classification_report(y_test, grid_predictions))
    accuracy = accuracy_score(y_test, grid_predictions)
    print("Predicciones:",grid_predictions)
    print("Real:", y_test)
    print("Precisión test:", accuracy)

    generar_excel(grid_predictions)






#Generamos EXCEL con los resultados
def generar_excel(predictions):
    df_y_test = pd.DataFrame(y_test)
    df_y_test.to_excel('y_test_h0_rf.xlsx', index=False, header=True)
    
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_excel('predictions_h0_rf.xlsx', index=False, header=True)
    
    
    
    
    
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