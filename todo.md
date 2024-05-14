
## Preguntas
- Mirar si cambiar como se indican las labels en los csv datasets, en la libreria de java estan distintas.
https://github.com/kdis-lab/MIML/blob/master/mavenProject/src/main/java/miml/data/MIMLInstances.java -> 
https://github.com/tsoumakas/mulan/blob/master/mulan/src/main/java/mulan/data/MultiLabelInstances.java
- Ver si hacer funciones a parte para load_cada_dataset y no tener que llamar a pkg_resources que queda feo, se
  puede mirar como esta implementado en https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_base.py, esta ya implementado en dataset_utils.py
- Llamar evaluate fuera report pasandoselo al init, si hago esto tengo que pasar dataset.get_features_by_bag(), 
  resultado_evaluate y nombre_etiquetas. Tampoco podria pasarle detalles como el clasificador, algoritmo y transformacion usada
- sk multilearn brknn, mlknn. Lo he probado pero no funciona, parece que esta desactualizada la libreria
- split dataset con bolsas en vez de con instancias
- A la hora de ejecutar un label powerset classifier, los clasificadores mi que hay implementados solo son capaces de 
  realizar una clasificacion binaria. Probar a incorporar CitationKNN https://github.com/arjunssharma/Citation-KNN/tree/master


### Documentacion
- Revisar todos los nombres

### Data

### Datasets
- Tener todos dataset en arff y csv

### Classifiers
- Que el predict_bag de ml devuelva un array solo, y ver el tipo con el que devuelve las labels
- Arreglar predict_proba MILESClassifier

### Report
- Arreglar la metrica de la curva roc, creo que al hacer el particionado del dataset pseudo-aleatorio se arreglaría, 
  no se arregla.

### Tutorials
- tutorial particionado datasets con varios clasificadores y sus reports
- Ejecutarlos todos de nuevo

### Otros
- Añadir info a pyproject.toml

