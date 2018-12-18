# CSUNSA-Seminario-de-Tesis-Codigo-Fuente
Codigo fuente de la propuesta de la tesis

Base de Datos:
la base de datos de tweets Visual Sentiment Ontology de la Universidad de Columbia (http://www.ee.columbia.edu/ln/dvmm/vso/download/twitter_dataset.html). Cada registro esta conformado por una imagen, un texto y el sentimiento relacionado al registro (positivo o negativo).

El codigo se divide en dos modulos:
- Analisis de Textos
  En este modulo contiene el codigo para el analisis de sentimientos en texto mediante el algoritmo de Distributed Paragraph Vectors. Se construye un modelo de entrenamiento que recibe los textos como vectores de entrenamiento para poder predecir palabras relacionadas tanto positivas como negativas. Los resultados del nos indican que tuvo una precisión de 0.75
  
  ![alt text](https://github.com/diegodrg27/CSUNSA-Seminario-de-Tesis-Codigo-Fuente/blob/master/Analisis%20de%20Texto/testing%20result.PNG)
  
- Analisis de Imagenes
  En este módulo contiene el codigo para el analisis de sentimientos en imagenes empleando una red neuronal en keras y empleando objectness score.
  
  En cuanto a la red neuronal el modelo fue construido y entrenado en Keras. Las imagenes se separaron 70% para entrenamiento y 30% para testing, cada una de ellas fue redimensionado en tamaños de 100x100, por lo que afectó los resultados del entrenamiento y validación. En cuanto a los resultados se obtuvo una precisión de entrenamiento de 0.725 y validación 0.87. 
  
  ![alt text](https://github.com/diegodrg27/CSUNSA-Seminario-de-Tesis-Codigo-Fuente/blob/master/Analisis%20de%20Imagen/resultados/accuracy.png)

![alt text](https://github.com/diegodrg27/CSUNSA-Seminario-de-Tesis-Codigo-Fuente/blob/master/Analisis%20de%20Imagen/resultados/sam3.png)
