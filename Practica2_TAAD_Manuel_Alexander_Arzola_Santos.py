from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')


# Función de menú, sirve para que el usuario seleccione de entre las diferentes opciones disponibles.
def menu(documentos):

    # El siguiente bucle while funciona como menú
    salir = False
    while not salir:
        print('                                ')
        print('     ######################     ')
        print('                                ')
        print('Seleccione una de las siguientes opciones:')
        print('1. Cargar nuevo archivo de documentos.')
        print('2. Visualizar los documentos cargados.')
        print('3. Visualizar tabla con los valores tf, idf, tfidf de los diferentes documentos.')
        print('4. Visualizar tabla de similitud entre los diferentes documentos.')
        print('5. Recibir una recomendación.')
        print('6. Salir.')
        print('                                ')
        print('     ######################     ')
        print('                                ')
        opcion = input()
        if opcion == '1':
            documentos = resenyas()
        elif opcion == '2':
            for elem in documentos:
                print(elem)
        elif opcion == '3':
            visualizar_tabla(documentos)
        elif opcion == '4':
            visualizar_similitud(documentos)
        elif opcion == '5':
            recomendacion(documentos)
        elif opcion == '6':
            salir = True
        else:
            print('Introduzca un número entre 1 y 6.')


# Función que permite leer el archivo con la base de datos de reseñas.
def resenyas():

    print('Introduzca la ruta del archivo que contiene los textos en formato txt')
    textos = input()

    f = open(textos, 'r', encoding="utf-8")
    mensaje = f.read()
    f.close()
    mensaje2 = mensaje.splitlines()

    return mensaje2


# Devuelve una matriz con los valores tfidf, otra con los valores tf y un vector
# con los valores de idf de cada uno de los términos.
def hallar_valores_todos(fuente):
    # Definimos las stopwords que va a usar la función TfidfVectorizer para eliminar
    # palabras comunes que no aportan nada.
    palabras = set(stopwords.words('english'))

    # Tokenizamos, construimos el vocabulario y hallamos los valores de tfidf para
    # cada término. Luego, lo convertimos en dataframe.
    vectorizer = TfidfVectorizer(analyzer='word', stop_words=palabras, use_idf=True)
    vector = vectorizer.fit_transform(fuente)

    tfidf = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names())

    # Convertimos el atributo .idf_, que nos devuelve la función anterior, en un vector.
    idf = pd.Series(data=vectorizer.idf_, name='Idf', index=vectorizer.get_feature_names())

    # Para calcular la matriz de valores tf, calculamos la inversa de cada uno de
    # los valores del vector idf, para así multiplicarla vectorialmente por la
    # matriz de valores tfidf. De esta forma obtendremos la matriz de valores tf.
    array_idf = 1/vectorizer.idf_
    array_tfidf = np.array(tfidf)
    array_tf = np.multiply(array_tfidf, array_idf)
    tf = pd.DataFrame(array_tf, columns=vectorizer.get_feature_names(), index=tfidf.index)

    return tfidf, idf, tf


def hallar_valores_tfidf(fuente):
    # Definimos las stopwords que va a usar la función TfidfVectorizer para eliminar
    # palabras comunes que no aportan nada
    palabras = set(stopwords.words('english'))

    # Tokenizamos, construimos el vocabulario y hallamos los valores de tfidf para
    # cada término. Luego, lo convertimos en dataframe.
    vectorizer = TfidfVectorizer(analyzer='word', stop_words=palabras, use_idf=True)
    vector = vectorizer.fit_transform(fuente)

    tfidf = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names())

    return tfidf


# Función que nos devuelve una tabla con las siguientes columnas:
# tf, idf, tfidf e índice del término, para cada una de las palabras y documentos.
def visualizar_tabla(documentos):

    # Llamamos a la función que nos devuelve los valores de tf, idf y tfidf.
    tfidf, idf, tf = hallar_valores_todos(documentos)

    # Pedimos al usuario que nos indique de qué documento desea ver los datos.
    salir = False
    while not salir:
        print("Introduzca el número de documento del que desea ver los datos.")
        print(f"El número de documentos es: {tfidf.shape[0]}.")
        documento = input()
        documento = int(documento) - 1
        if 0 <= documento <= tfidf.shape[0] - 1:
            salir = True
        else:
            print(f'Introduzca un número entre 1 y {tfidf.shape[0]}.')
            print('                                ')

    # Agregamos al dataframe tfidf una nueva fila con los valores idf.
    tfidf_nuevo = tfidf.append(idf, ignore_index=False)

    # Creamos un nuevo dataframe sólo con las filas que nos interesan,
    # la de los valores tfidf del documento deseado y la de los valores idf.
    tfidf_sub = tfidf_nuevo.iloc[[documento, -1]]

    # Seleccionamos la fila de la matriz tf con los valores del documento
    # deseado y la añadimos al dataframe anterior.
    tf_sub = tf.iloc[[documento]]
    inter = tfidf_sub.append(tf_sub, ignore_index=False)

    # Creamos una serie que tenga los índices para cada uno de los términos
    # y la añadimos al dataframe anterior.
    indices = pd.Series(data=np.arange(tfidf.shape[1]), name='Indice', index=idf.index)
    final = inter.append(indices, ignore_index=False)

    # Cambiamos los nombres de las filas.
    final.set_axis(['TfIdf', 'Idf', 'Tf', 'Indice'], axis='index', inplace=True)

    # Creamos una lista de los términos que no aparecen en el documento,
    # para eliminarlos en un futuro y quedarnos sólo con las palabras que aparecen en el documento en cuestión.
    lista = []

    for elem in final.columns:
        if final.iloc[0][elem] == 0:
            lista.append(elem)

    final.drop(lista, axis=1, inplace=True)

    # Trasponemos el dataframe para una mejor visualización.
    prin = final.T
    prin['Indice'] = prin['Indice'].astype('int64')

    print(prin)


# Función que nos permite ver la similitud entre los diferentes
# documentos de la base de datos de reseñas.
def visualizar_similitud(documentos):

    # Llamamos a la función que nos devuelve los valores tfidf.
    tfidf = hallar_valores_tfidf(documentos)

    # Creamos un nuevo dataframe donde almacenar los valores de similitud.
    df = pd.DataFrame()

    # Calculamos el vector de similitud entre cada par de documentos
    # y lo almacenamos en el dataframe.
    for elem in range(tfidf.shape[0]):
        variable = cosine_similarity(tfidf[elem:elem + 1], tfidf)
        variable2 = pd.DataFrame(variable)
        df = df.append(variable2)

    # Renombramos los nombres de filas y columnas para una mejor visualización.
    for elem in df.columns:
        df.rename(columns={elem: ('Doc ' + str(int(elem) + 1))}, inplace=True)
    df.set_axis(df.columns, axis='index', inplace=True)

    print(df)


# Función que nos recomienda un documento de entre los diferentes almacenados
# en base a lo similar que sea con otro documento diferente que le introduzcamos.
def recomendacion(textos):
    # Llamamos a la función que nos permite leer un archivo de texto externo.
    texto = resenyas()

    # Añadimos el nuevo documento al vector de documentos previo.
    textos.extend(texto)

    # Llamamos a la función que nos devuelve los valores tfidf.
    matriz_tfidf = hallar_valores_tfidf(textos)

    # Calculamos la similitud del coseno entre el documento nuevo y el resto de documentos.
    # Luego, creamos dos listas que contienen los valores ordenados y sin ordenar.
    coseno = cosine_similarity(matriz_tfidf.tail(1), matriz_tfidf.drop(matriz_tfidf.index[[-1]]))
    coseno_sub = coseno[0]
    coseno_list = coseno_sub.flatten().tolist()
    coseno_list_sort = sorted(coseno_list, reverse=True)

    # Variables de control del bucle que nos mostrará los documentos más cercanos.
    # La variable tope indica el número de documentos más cercanos que será mostrado.
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


if __name__ == '__main__':

    texto = resenyas()

    menu(texto)

    # Mensaje de despedida
    print('                                ')
    print('Cerrando, que tenga un buen día.')
