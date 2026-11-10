# Machine-Learning-for-Tourist-Accommodation

Machine Learning Application for Tourist Accommodation: Prediction of Hotel Booking Cancellations and Average Daily Rate. (From now everything in Spanish)

Trabajo de Fin de Grado 2020/2021 - Universidad Complutense de Madrid - Grado en Turismo

## Sobre el proyecto

Este repositorio conserva el trabajo desarrollado para mi **Trabajo de Fin de Grado en Turismo** en la Universidad Complutense de Madrid, durante el curso **2020/2021**, bajo la tutoría de **José Carlos Soto Gómez**. La calificación obtenida fue **8,9**.

Quise explorar cómo podían aplicarse técnicas de *Machine Learning* a problemas reales del sector turístico. Era mi **primera experiencia práctica con Python y aprendizaje automático**, de modo que el proyecto supuso tanto una investigación académica como un proceso de aprendizaje desde cero. En 2021 estas herramientas todavía no tenían la presencia pública que alcanzarían años después.

El trabajo completo se titula **«Aplicación de Machine Learning para Alojamientos Turísticos: Predicción de Cancelación de Reservas Hoteleras y de la Tarifa Media Diaria»**. La memoria original —con marco teórico, estado del arte, metodología, gráficas, resultados, conclusiones, bibliografía y anexos— se conserva íntegra aquí:

**[Leer la memoria completa del TFG (PDF)](Desarrollo%20del%20Proyecto/Aplicaci%C3%B3n%20de%20Machine%20Learning%20para%20Alojamientos%20Tur%C3%ADsticos_%20Predicci%C3%B3n%20de%20Cancelaci%C3%B3n%20de%20Reservas%20Hoteleras%20y%20de%20la%20Tarifa%20Media%20Diaria.pdf)**

Este README es una guía para orientarse por el repositorio, **no sustituye a la memoria**.

## Qué se investigó

El objetivo principal era mostrar posibles aplicaciones del aprendizaje automático en alojamientos turísticos. Se trabajó en tres líneas:

1. **Cancelaciones de reservas:** predecir si una reserva se cancelaría, tanto para el conjunto completo como por separado para un hotel urbano y uno de tipo resort.
2. **Tarifa media diaria (ADR):** estimar el precio diario asociado a una reserva.
3. **Análisis de datos:** estudiar relaciones entre variables y segmentos de mercado mediante análisis exploratorio, comparaciones bivariadas, análisis multivariado y componentes principales (PCA).

La memoria también desarrolla el contexto turístico y tecnológico: evolución del turismo, transformación digital, Big Data, inteligencia artificial y aplicaciones del aprendizaje automático en demanda turística, recomendaciones, opiniones de clientes y cancelaciones.

## Datos utilizados

Se partió del conjunto de datos de reservas hoteleras publicado por **Antonio, Almeida y Nunes (2019)**, disponible también en Kaggle. Contiene **119.390 reservas** de dos hoteles reales de Portugal: uno de tipo resort en el Algarve (**40.060**) y otro urbano en Lisboa (**79.330**).

El conjunto original tenía **32 variables**. Tras tratar los valores ausentes —entre otras operaciones, sustituyendo los campos vacíos de agente y empresa por cero y descartando registros incompletos en otros campos— se trabajó con **118.898 reservas**. Después se transformaron categorías en valores numéricos, se trataron fechas y se prepararon diferentes conjuntos de variables para los modelos.

La preparación no fue idéntica en todos los experimentos: se seleccionaron variables según el objetivo y el tipo de hotel. Los CSV intermedios y finales se conservan en sus respectivas carpetas.

## Cómo se desarrolló

El trabajo siguió una planificación documentada en la memoria, con un **diagrama BPMN** y un **diagrama de Gantt**. El desarrollo práctico se organizó aproximadamente así:

1. **Exploración y limpieza:** revisión de tipos de datos, valores nulos, distribuciones y registros; filtrado y transformación de variables.
2. **Análisis exploratorio:** gráficas y comparaciones entre variables, con atención al segmento de mercado.
3. **Análisis multivariado:** normalización y aplicación de PCA para representar información en menos dimensiones. En el análisis recogido en la memoria se retuvo un **51 % de la varianza**.
4. **Selección de variables:** matrices de correlación e importancia de variables para preparar los datasets de cada predicción.
5. **Entrenamiento y comparación:** modelos de clasificación y regresión, con pruebas de parámetros y validación.
6. **Evaluación:** comparación de valores reales y predichos, métricas y exportación de resultados a Excel.

### Modelos y herramientas

- **Cancelación:** máquinas de vectores de soporte (**SVM**) y **Random Forest**.
- **ADR:** regresión mediante vectores de soporte (**SVR**, implementada con `LinearSVR`) y **Random Forest Regressor**.
- **Evaluación:** separación habitual del **80 % para entrenamiento y 20 % para pruebas**; validación cruzada y búsquedas de parámetros en distintos experimentos; matrices de confusión, precisión, *recall*, F1 y medidas de error.
- **Entorno de la época:** Python **3.8.3**, Anaconda/Spyder, pandas, NumPy, Matplotlib, Seaborn y scikit-learn.

Los scripts son los originales del proyecto y reflejan las herramientas y versiones utilizadas entonces.

## Resultados obtenidos en el TFG

Estos son los **resultados históricos comunicados en la memoria de 2021**, no una reevaluación actual de los modelos:

| Problema | Resultado recogido en la memoria |
| --- | --- |
| Cancelación de reservas | Hasta aproximadamente **94 % de acierto** con Random Forest en los escenarios analizados |
| Cancelación, conjunto completo | **1.323 errores** en **23.780** reservas de prueba con Random Forest, frente a **1.573** con SVM |
| Predicción de ADR con Random Forest | **78 %** de predicciones dentro de un margen de **±10 €** respecto al valor real |
| Predicción de ADR con SVR | **30,87 %** dentro de ese mismo margen |

En la memoria se presentan también cifras de error para ADR (7,2 y 15,6 para Random Forest, frente a aproximadamente 26 y 39 para SVR). Conviene consultar allí el contexto exacto de esas métricas; **MAE y RMSE se calculan en las unidades de la variable predicha**, por lo que no deben interpretarse automáticamente como porcentajes.

La conclusión académica fue que **Random Forest ofrecía los mejores resultados** entre los modelos probados, tanto en clasificación como en regresión. El documento también identifica mejoras posibles que quedaron fuera del alcance del TFG.

## Organización del repositorio

| Carpeta | Contenido |
| --- | --- |
| [Desarrollo del Proyecto](Desarrollo%20del%20Proyecto/) | **Memoria académica completa** en PDF |
| [Esquema BPMN del proyecto](Esquema%20BPMN%20del%20proyecto/) | Diagrama original del proceso (`.bpmn`) |
| [Primer Análisis de datos](Primer%20An%C3%A1lisis%20de%20datos/) | Scripts de consulta, limpieza, transformación y tratamiento inicial de fechas |
| [Comparación univariada y PCA](Comparaci%C3%B3n%20univariada%20y%20PCA/) | Carpeta histórica; el análisis correspondiente se explica en la memoria |
| [Predecir Cancelación Hotel Resort](Predecir%20Cancelaci%C3%B3n%20Hotel%20Resort/) | Preparación de variables, SVM, Random Forest y resultados del resort |
| [Predecir Cancelación Hotel Urbano](Predecir%20Cancelaci%C3%B3n%20Hotel%20Urbano/) | Preparación de variables, SVM, Random Forest y resultados del hotel urbano |
| [Predecir Cancelación Dataset Completo](Predecir%20Cancelaci%C3%B3n%20Dataset%20Completo/) | Experimentos de cancelación con ambos hoteles |
| [Predecir ADR](Predecir%20ADR/) | Experimentos de predicción de tarifa media diaria con SVR y Random Forest |
| [Resultados Finales Predicción Consolidados](Resultados%20Finales%20Predicci%C3%B3n%20Consolidados/) | Archivos históricos de resultados consolidados |

En las carpetas de predicción aparecen scripts como `dataset_*.py`, `correlation_*.py`, `variable_importance_*.py`, `SVM_*.py` y `random_forest_*.py`. Los directorios `Resultados` contienen hojas Excel con predicciones, valores reales y variantes de experimentos. Se mantienen también los CSV y archivos auxiliares originales, incluso cuando se repiten entre carpetas.

## Alcance y conservación histórica

Este repositorio es un **archivo de un proyecto académico realizado en 2020/2021**, no una aplicación mantenida para uso hotelero en producción. No se han modernizado ni reescrito los experimentos para esta documentación.

Hay algunas limitaciones que conviene conocer al interpretar los resultados:

- En los modelos de cancelación se eliminó `reservation_status`, pero se conservó `reservation_status_date`. Esta fecha puede reflejar información conocida **después** de la reserva o de su cancelación. Por ello, el porcentaje de acierto documentado **no demuestra por sí solo** que se alcanzaría la misma precisión al predecir cancelaciones futuras con datos disponibles en el momento de la reserva.
- La selección de variables y algunas transformaciones se realizaron sobre los datos antes de separar entrenamiento y prueba; una reevaluación moderna debería revisar ese procedimiento.
- El código depende de las versiones de Python y bibliotecas de la época y puede requerir adaptaciones para ejecutarse hoy.
- Los resultados recogidos aquí proceden de la memoria y de los archivos históricos; **no se han vuelto a entrenar los modelos** para esta documentación.

Estas observaciones sirven para contextualizar el trabajo, **no para modificar retrospectivamente lo realizado**. La memoria, el código, los datasets, los diagramas y los resultados se conservan como parte del proyecto original.

## Referencia principal

La explicación detallada de cada decisión, el **marco teórico**, el **estado del arte**, la **metodología**, los análisis, las figuras, las tablas, las **conclusiones** y las **referencias bibliográficas** están en la [memoria original del TFG](Desarrollo%20del%20Proyecto/Aplicaci%C3%B3n%20de%20Machine%20Learning%20para%20Alojamientos%20Tur%C3%ADsticos_%20Predicci%C3%B3n%20de%20Cancelaci%C3%B3n%20de%20Reservas%20Hoteleras%20y%20de%20la%20Tarifa%20Media%20Diaria.pdf). Es la fuente principal para entender y citar el proyecto.
