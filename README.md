# Técnicas avanzadas de análisis de datos. Proyecto sobre sistemas recomendadores basados en contenido.

El objetivo de este proyecto es implementar un sistema de recomendación basado en contenido. Para ello se ha realizado un programa en python, el cual usa las siguientes funciones, que paso a detallar.

Todas las funciones excepto `resenyas()` requieren como argumento de entrada la variable que contiene la base de datos de documentos.

## Función `menu(documentos)`

La usaremos para que el usuario seleccione de entre las diferentes opciones disponibles, las cuales son las siguientes:

1. Cargar nuevo archivo de documentos.
2. Visualizar los documentos cargados.
3. Visualizar tabla con los valores tf, idf, tfidf de los diferentes documentos.
4. Visualizar tabla de similitud entre los diferentes documentos.
5. Recibir una recomendación.
6. Salir.

Al introducir el número perteneciente a la opción deseada, se llamará a la función correspondiente.


## Función `resenyas()`

Sencilla función que pide por consola la ruta del archivo que contiene el conjunto de documentos. Cada una de los documentos debe estar en una única línea.
Abre el archivo, separa cada una de las líneas del archivo y devuelve un vector cuyos elementos son los documentos.


## Función `hallar_valores_todos(fuente)`

Devuelve una matriz con los valores tfidf, otra con los valores tf y un vector con los valores de idf de cada uno de los términos. Para ello, usaremos la función `TfidfVectorizer`, la cual pertenece a la librería scikit learn. 

Primero, definimos las stopwords que va a usar, para eliminar palabras comunes que no aportan nada. Acto seguido, tokenizamos, construimos el vocabulario y hallamos los valores de tfidf para cada término. Luego, lo convertimos en dataframe.
```
    vectorizer = TfidfVectorizer(analyzer='word', stop_words=palabras, use_idf=True)
    vector = vectorizer.fit_transform(fuente)
```
A continuación, hallamos el vector de valores idf, el cual nos es proporcionado como atributo de la salida de la función `TfidfVectorizer`. Lo convertimos a una serie con la librería pandas.
```
    idf = pd.Series(data=vectorizer.idf_, name='Idf', index=vectorizer.get_feature_names())
```
Por último, obtenemos la matriz de valores tf. Estos valores no nos los proporciona `TfidfVectorizer`, por lo que debemos hallarlos manualmente. Para ello, calculamos la inversa de cada uno de los valores del vector idf, para así multiplicarla vectorialmente por la matriz de valores tfidf.
```
    array_idf = 1/vectorizer.idf_
    array_tfidf = np.array(tfidf)
    array_tf = np.multiply(array_tfidf, array_idf)
    tf = pd.DataFrame(array_tf, columns=vectorizer.get_feature_names(), index=tfidf.index)
```


## Función `hallar_valores_tfidf(fuente)`

Esta función funciona de manera análoga a la anterior, con la salvedad de que sólo realiza las operaciones concernientes a la matriz de valores tfidf.


## Función `visualizar_tabla(documentos)`

Función que nos devuelve una tabla con las siguientes columnas: tf, idf, tfidf e índice del término, para cada una de las palabras y documentos. Empezamos llamando a la función `hallar_valores_todos(fuente)`, que nos devuelve los valores tf, idf y tfidf. 

Luego, le mostramos al usuario el número de documentos disponibles y le pedimos que elija uno.

Agregamos al dataframe tfidf una nueva fila con los valores idf, y eliminamos el resto de filas excepto la de idf y la del documento que nos interesa. También seleccionamos del dataframe tf el relativo al documento que nos interesa y lo añadimos al dataframe previo.

Por último, creamos una serie que tenga los índices para cada uno de los términos y la añadimos al dataframe anterior, con esto tendremos un dataframe con los valores a mostrar.

Tan sólo nos resta eliminar las palabras que no aparezcan en el documento. Creamos una lista de los términos que no aparecen en el documento, para eliminarlos y quedarnos sólo con las palabras que aparecen en el documento en cuestión.
```
    lista = []

    for elem in final.columns:
        if final.iloc[0][elem] == 0:
            lista.append(elem)

    final.drop(lista, axis=1, inplace=True)
```
Finalmente, trasponemos el dataframe para una mejor visualización y lo imprimimos por pantalla.


## Función `visualizar_similitud(documentos)`

Función que nos permite ver la similitud entre los diferentes documentos de nuestro conjunto. Llamamos a la función `hallar_valores_tfidf(fuente)`, que nos devuelve los valores tfidf. Luego, creamos un dataframe donde almacenamos los valores de similitud. A continuación calculamos el vector de similitud entre cada par de documentos y lo almacenamos en el dataframe.
```
    for elem in range(tfidf.shape[0]):
        variable = cosine_similarity(tfidf[elem:elem + 1], tfidf)
        variable2 = pd.DataFrame(variable)
        df = df.append(variable2)
```
Por último, renombramos los nombres de filas y columnas para una mejor visualización e imprimimos el dataframe por pantalla.
```
    for elem in df.columns:
        df.rename(columns={elem: ('Doc ' + str(int(elem) + 1))}, inplace=True)
    df.set_axis(df.columns, axis='index', inplace=True)

    print(df)
```


## Función `recomendacion(textos)`

Función que nos recomienda un documento de entre los diferentes almacenados en base a lo similar que sea con otro documento diferente que le introduzcamos.

Primero, llamamos a la función que nos permite leer un archivo de texto externo, para introducir el nuevo documento; lo añadimos al vector de documentos previo y llamamos a la función `hallar_valores_tfidf(fuente)`, que nos devuelve los valores tfidf de los documentos.

Calculamos la similitud del coseno entre el documento nuevo y el resto de documentos y creamos dos listas que contienen los valores ordenados y sin ordenar.
```
    coseno = cosine_similarity(matriz_tfidf.tail(1), matriz_tfidf.drop(matriz_tfidf.index[[-1]]))
    coseno_sub = coseno[0]
    coseno_list = coseno_sub.flatten().tolist()
    coseno_list_sort = sorted(coseno_list, reverse=True)
```
Con el siguiente bucle, mostramos por pantalla los n elementos más similares al documento introducido, donde n = tope:
```
    tope = 5
    h = 0

    print(' ')
    print(f"Se mostrarán los {tope} elementos con mayor similitud:")
    for elem in coseno_list_sort:
        if h < tope:
            print(' ')
            print(f"La similitud con el Documento {coseno_list.index(elem) + 1} es: {elem}.")
            print(f"El texto es: {textos[coseno_list.index(elem)]}.")
            h = h+1
```

## Función principal o main

En la función principal tan sólo tenemos la llamada a la función `resenyas()` para introducir la base de documentos inicial, y la llamada a la función `menu(documentos)`, mediante la cual controlaremos todo el programa.

Para finalizar, el programa nos mostrará un mensaje de despedida.