# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:12:54 2021

@author: alvar
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


#Leemos los datos (20)
df = pd.read_csv("dataset_final_h1.csv")



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
    clf = svm.SVC(gamma='scale', class_weight='balanced')
    # clf = svm.SVC(C=100, gamma=0.0001, class_weight='balanced') #Optimizado
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




def optimizado():

    #Defino parámetros para SVM. GridSearch los optimizará luego
    #Originalmente: C: 0.1, 1, 10, 100, 1000;   gamma: 1, 0.1, 0.01, 0.001, 0.0001
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']} 
    
    #Indico modelo, parámetros, refit=True para optimizar parámetros
    grid = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid, refit=True, verbose=10)
    
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
    df_y_test.to_excel('y_test_h1_svm.xlsx', index=False, header=True)
    
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_excel('predictions_h1_svm.xlsx', index=False, header=True)
    
    
    
    
    
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