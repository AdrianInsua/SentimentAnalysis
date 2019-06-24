"""
Auxiliar para la función de composición del corpus de datos

"""

# Autor Adrián Insua Yañez
import pandas as pd
import os
from Corpus import XmlDataFrame

from lxml import objectify

# Display progress logs on stdout
pd.options.mode.chained_assignment = None


def get_corpus(inp, output, v, comments=None, not_clasif=None):
    """

    :param inp: Dirección del fichero de datos. String
    :param output: Dirección en la que se guardarán los datos. String
    :param v: Nivel de verbose. int
    :param comments: comentarios clasificados
    :param not_clasif: comentarios no clasificados
    :return: data_corpus. Instancia de pandas DataFrame

    Primero intenta recuperar el corpus del archivo csv persistido en disco con el nombre del input
    En caso de no encontrar el fichero parsea el xml con el nombre especificado en el input

    """
    # cine = MuchoCineGetter(10, "http://www.muchocine.net/criticas_ultimas.php", 2)
    print("#Intentando obtener datos del archivo csv...") if v >= 1 else None
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/'+inp+'.csv')
    print(filename)
    try:
        data_corpus = pd.read_csv(filename, encoding='utf-8')
    except Exception as e:
        # Si no existe csv se parsea el xml para crearlo
        print("Error: ", e,
              "#\n Parseando el xml...") if v >= 1 else None
        xml_filename = os.path.join(dirname, 'data/'+inp+'.xml')
        xml = objectify.parse(open(xml_filename))
        root = xml.getroot()

        data_frame_helper = XmlDataFrame.DataFrameHelper(root.getchildren(), output, v)
        data_corpus = data_frame_helper.convert()
        #data_corpus = cine.get_corpus()
        #data_corpus = data_corpus.append(c, ignore_index=True)
        data_corpus.polarity[data_corpus.polarity.isin(['N+'])] = 1
        data_corpus.polarity[data_corpus.polarity.isin(['N'])] = 2
        data_corpus.polarity[data_corpus.polarity.isin(['NEU'])] = 3
        data_corpus.polarity[data_corpus.polarity.isin(['P'])] = 4
        data_corpus.polarity[data_corpus.polarity.isin(['P+'])] = 5
        data_corpus.to_csv(filename, index=False, encoding='utf-8')

    # Impresiones verbose
    print("#Datos recuperados!") if v >= 1 else None
    print("Tamaño del corpus: ", len(data_corpus)) if v >= 2 else None
    print("Corpus: ", data_corpus) if v >= 3 else None
    return data_corpus


def filter_corpus_small(data_corpus):
    data_corpus.polarity[data_corpus.polarity.isin([1, 2])] = 1
    data_corpus.polarity[data_corpus.polarity.isin([3])] = 2
    data_corpus.polarity[data_corpus.polarity.isin([4, 5])] = 3
    return data_corpus


def filter_binary_pn(data_corpus):
    data_corpus.polarity[data_corpus.polarity.isin([1, 2])] = 0
    data_corpus.polarity[data_corpus.polarity.isin([4, 5])] = 1
    data_corpus = data_corpus[data_corpus.polarity != 3]
    return data_corpus.reset_index()


def filter_binary_neun(data_corpus):
    data_corpus.polarity[data_corpus.polarity.isin([1, 4, 5])] = 1
    data_corpus.polarity[data_corpus.polarity.isin([3])] = 0
    return data_corpus


def filter_binary_pp(data_corpus):
    data_corpus = data_corpus[data_corpus.polarity.isin([4, 5])]
    data_corpus.polarity[data_corpus.polarity == 4] = 1
    data_corpus.polarity[data_corpus.polarity == 5] = 0
    return data_corpus.reset_index()


def filter_binary_nn(data_corpus):
    data_corpus = data_corpus[data_corpus.polarity.isin([1, 2])]
    data_corpus.polarity[data_corpus.polarity == 1] = 0
    data_corpus.polarity[data_corpus.polarity == 2] = 1
    return data_corpus.reset_index()





