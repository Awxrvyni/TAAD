# Técnicas avanzadas de análisis de datos. Proyecto sobre sistemas recomendadores basados en contenido.

El objetivo de este proyecto es implementar un sistema de recomendación basado en contenido. Para ello se ha realizado un programa en python, el cual usa las siguientes funciones, que paso a detallar.



## Función `menu(documentos)`

La usaremos para que el usuario seleccione de entre las diferentes opciones disponibles, las cuales son las siguientes:

1. Cargar nuevo archivo de reseñas.
2. Visualizar tabla con los valores de tf, idf, tfidf de las diferentes reseñas.
3. Visualizar tabla de similitud entre las diferentes reseñas.
4. Recibir una recomendación.
5. Visualizar los documentos cargados.
6. Salir.

Al introducir el número perteneciente a la opción deseada, se llamará a la función correspondiente.

Esta función requiere como argumento de entrada la variable que contiene la base de datos de reseñas.



## Función `resenyas()`

Sencilla función que pide por consola la ruta del archivo que contiene la base de datos de reseñas. Cada una de las reseñas debe estar en una única línea.
Abre el archivo, separa cada una de las líneas del archivo y devuelve un vector cuyos elementos son las reseñas.


## Función `hallar_valores_todos(fuente)`

Devuelve una matriz con los valores tfidf, otra con los valores tf y un vector con los valores de idf de cada uno de los términos. Para ello, usaremos la función `TfidfVectorizer`, la cual pertenece a la librería scikit learn. Primero, definimos las stopwords que va a usar, para eliminar palabras comunes que no aportan nada. Acto seguido, tokenizamos, construimos el vocabulario y hallamos los valores de tfidf para cada término. Luego, lo convertimos en dataframe.
```
    vectorizer = TfidfVectorizer(analyzer='word', stop_words=palabras, use_idf=True)
    vector = vectorizer.fit_transform(fuente)
```
A continuación, hallamos el vector de valores idf, el cual nos es proporcionado como atributo de la salida de la función `TfidfVectorizer`. Lo convertimos a una serie con la librería pandas.
```
    idf = pd.Series(data=vectorizer.idf_, name='Idf', index=vectorizer.get_feature_names())
```
Por último, hallamos la matriz de valores tf. Estos valores no nos los proporciona `TfidfVectorizer`, por lo que debemos hallarlos manualmente. Para ello, calculamos la inversa de cada uno de los valores del vector idf, para así multiplicarla vectorialmente por la matriz de valores tfidf.
```
    array_idf = 1/vectorizer.idf_
    array_tfidf = np.array(tfidf)
    array_tf = np.multiply(array_tfidf, array_idf)
    tf = pd.DataFrame(array_tf, columns=vectorizer.get_feature_names(), index=tfidf.index)
```
Esta función requiere como argumento de entrada la variable que contiene la base de datos de reseñas.


## Función `hallar_valores_tfidf(fuente)`

Esta función funciona de manera análoga a la anterior, con la salvedad de que sólo devuelve la matriz de valores tfidf.


## Función `visualizar_tabla(documentos)`

Función que nos devuelve una tabla con las siguientes columnas: tf, idf, tfidf e índice del término, para cada una de las palabras y documentos. Empezamos llamando a la función `hallar_valores_todos(fuente)`, que nos devuelve los datos de tf, idf y tfidf de los documentos. 
Luego, le mostramos al usuario el número de documentos disponibles y le pedimos que elija uno.
Agregamos al dataframe tfidf una nueva fila con los valores idf, y eliminamos el resto de filas excepto la de idf y la del documento que nos interesa. También seleccionamos del dataframe tf el relativo al documento que nos interesa y lo añadimos al dataframe previo.
Por último, creamos una serie que tenga los índices para cada uno de los términos y la añadimos al dataframe anterior, con esto tendremos un dataframe con los valores a mostrar.
Tan sólo nos resta eliminar las palabras que no aparezcan en el documento. Creamos una lista de los términos que no aparecen en el documento, para eliminarlos y quedarnos sólo con las palabras que aparecen en el documento en cuestión.
````
    lista = []

    for elem in final.columns:
        if final.iloc[0][elem] == 0:
            lista.append(elem)

    final.drop(lista, axis=1, inplace=True)
````

